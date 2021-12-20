#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import abc
import requests
import traceback
import asyncio
from asyncio.events import AbstractEventLoop
import aiohttp
import itertools
import multiprocessing
from datetime import datetime
from sqlalchemy import create_engine
from typing import Any, List, Set, Tuple, Dict, Optional, Literal, Union
import argparse

import wrapt
import tqdm # type: ignore 
from dotenv import load_dotenv

from bs4 import BeautifulSoup  # type: ignore 
from bs4.element import Tag, ResultSet  # type: ignore

import pandas as pd
import numpy as np

from utils.parallelization import parallelize_df  # type: ignore
from utils.logger import _get_logger  # type: ignore 

logger = _get_logger(__name__)

load_dotenv()

BASE_URL = os.getenv('BASE_URL')
ASYNCIO_CHUNKSIZE = 500
RETRIES_SINGLE_PAGE = 10
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_USERNAME = os.getenv('POSTGRES_USERNAME')
POSTGRES_IP = os.getenv('POSTGRES_IP')

class _CommandLine:
    def __init__(self):
        parser = argparse.ArgumentParser(description="")
        parser.add_argument("-i", "--industry", help="Optional", required=False, default=None)
        argument = parser.parse_args()

        self.industry = argument.industry
        print("Selected industry: {0}".format(argument.industry))



def parse_date(series):
    """Parses dates and fills NaNs with today."""
    return (pd.to_datetime(series, format="%d.%m.%y", errors="coerce")
                    .fillna(pd.to_datetime(datetime.now().date())))


async def asyncget_url_data(url, loop) -> Tuple[str, str]:
    """
    Download html data as str for given url.

    TODO: Retries indefinitely on server errors.

    Parameters
    ----------
    url: str
    loop: EventLoop (from `asyncio.get_event_loop`)

    Return
    ------
    str
        url equal to parameter
    str
        html data as string

    """

    html: str
    success: bool = False
    ntry: int = 0


    # print(f"                asyncget_url_data: get url {url}")

    while not success:
        try:
            ntry += 1
            async with aiohttp.ClientSession(loop=loop) as client:
                async with client.get(url=url) as resp:

                    html = await resp.text()

            if ntry > 2:
                logger.warning(f"                {ntry - 1} retries to fetch "
                               f"{url} with status {resp.status}")

            success = True

        except Exception as e:
            if ntry == RETRIES_SINGLE_PAGE:
                logger.error(f"Reached maximum number of retries: {ntry}")
                raise e
            else:           
                logger.debug(e)  # we are retrying indefinitely

    return url, html


def _fetch_list(urls: List[str], progress_desc: Optional[str]=None) -> List[Tuple[str, str]]:
    '''
    Returns list of html strings for list of urls.

    Tries asyncio. Falls back to serial.

    Parameters
    ----------
    urls: List[str]
        List of URLs
    progress_desc: str
        Tqdm progress description
          
    Returns
    -------
    List[Tuple[str, str]]
        List of tuples (url, html text)

    '''

    logger.info(f"        Retrieving list of urls (n={len(urls)}): {progress_desc}...")

    list_html: List[Tuple[str, str]]

    try:
        loop: AbstractEventLoop = asyncio.get_event_loop()

        list_html = loop.run_until_complete(
            asyncio.gather(*[asyncget_url_data(url, loop)
                             for url in urls]))

        logger.info("        ... done retrieving list of urls using asyncio.")

    except RuntimeError as e:
        logger.warning(f'        Async requests failed with RuntimeError: "{e}". '
                       'Falling back to sequential requests.')

        urls_prog = tqdm.tqdm(urls)
        urls_prog.set_description(desc=progress_desc)

        list_html = []

        url: str
        for url in urls_prog:
            list_html.append((url, requests.get(url).text))

        logger.info("        ... done retrieving list of urls using sequential requests.")

    except Exception as e:
        logger.error("        Unexpected error in _fetch_list")
        logger.warning(traceback.format_exc())
        raise e

    return list_html


def _fetch_chunked_list(urls: List[str], as_string: bool=False, 
               progress_desc: Optional[str]=None) -> List[Tuple[str, Union[BeautifulSoup, str]]]:
    """
    Call asynchronous functions to retrieve html text and generate BeautifulSoup objects.

    Splits list of URLS into chunks of approximate size ASYNCIO_CHUNKSIZE.

    Parameters
    ----------
    urls: List[str]
        list of URLs
    as_string: bool
        if False (default), return BeautifulSoup object; if True, return html as string
    progress_desc: Optional[str]
        tqdm progress description

    Returns
    -------
    List[Tuple[str, BeautifulSoup]]
        List of tuples (url, BeautifulSoup object)

    """

    list_html: List[List[Tuple[str, str]]]
    result_html: List[Tuple[str, str]]
    result_soup: List[Tuple[str, BeautifulSoup]]

    len_urls: int = len(urls)
    nchunks: int = len_urls // ASYNCIO_CHUNKSIZE

    logger.info(f"    Number of chunks for async requests: {max(1, nchunks)} for {len_urls} urls")

    urls_chunks: List[List[str]] = list(np.array_split(list(urls), nchunks)) if nchunks > 1 else [urls]

    logger.info(f"    _fetch_chunked_list: splitted url list into {len(urls_chunks)} "
                f"chunks with lengths {[len(chunk) for chunk in urls_chunks]}")

    list_html = [_fetch_list(urls_chunk, progress_desc) 
                 for urls_chunk in urls_chunks]

    result_html = list(itertools.chain.from_iterable(list_html))

    if as_string:
        return result_html

    else:
        logger.info(f"    Instantiating soup list ({progress_desc})...")
        result_soup = [(url, BeautifulSoup(html, 'html.parser')) 
                       for url, html in result_html]
        logger.info("    ... done instantiating soup list.")

        return result_soup


def get_industries(drop_zero: bool=True) -> Dict[str, str]:
    '''
    Retrieves a list of industries from BASE_URL/aktien/aktien_suche.asp

    Reason: Server won't return more than 1000 stocks per search request...
    need to filter *somehow*.

    Parameters
    ----------
    drop_zero: bool
        If True, removes the key "0" from the result dictionary corresponding
        to all stocks ("Alle", unfiltered).

    Returns
    -------
    Dict[str, str]
        Dictionary industry index => industry name

    '''

    url: str = f'{BASE_URL}/aktien/aktien_suche.asp'
    text: str = requests.get(url).text
    soup: BeautifulSoup = BeautifulSoup(text, 'html.parser')
    select_industry: Tag = soup.find("select", {"name": "inBranche"})
    options: ResultSet = select_industry.find_all('option')
    dict_industry_id: Dict[str, str] = {option['value']: option.text
                                        for option in options}

    if drop_zero:
        dict_industry_id.pop('0')  # Drop "Alle"

    return dict_industry_id


def gen_url_by_industry(ind: str, page: int) -> str:
    '''
    Generates url from industry id and page number.

    Parameters
    ----------
    ind: str
         Index of an industrial branch as listed in the keys
         of the dictionary returned by `get_industries()`
    page: int
         Page of the filtered result list.

    Returns
    -------
    str
        URL of filtered company list

    '''

    url:str = (
        f"{BASE_URL}/aktien/aktien_suche.asp?"
        f"intpagenr={page}&inbranche={ind}&"
        "inindex=0&infunndagrkl1=2&infunndagrkl2=2&inland=0&"
        "inbillanz=0&inbillanzjahr=2019&inbillanzgrkl=2&"
        "infundamental1=0&infundamentaljahr1=2019&"
        "infundamental2=0&infundamentaljahr2=2019&"
        "insonstige=0&insonstigegrkl=2")

    return url


def get_page_num(soup: BeautifulSoup) -> int:
    """
    Extract number of pages from BeautifulSoup list page.

    Parameters
    ----------
    soup: BeautifulSoup

    Return
    ------
    int
        Number of pages
    """

    try:
        n_page: int = int(soup.find('a', {'class': 'last'}).text)
    except AttributeError as e:
        n_page = 1

    return n_page


def get_first_pages(dict_industries: Dict[str, str]) -> List[Tuple[str, BeautifulSoup]]:
    '''
    Get first pages of company lists of all industries.
    This enables retrieval of page numbers. Soups are re-used later.

    Parameters
    ----------
    dict_industries: Dict[str, str]
        dictionary index => name of industry branch
    
    Returns
    -------
    List[Tuple[str, BeautifulSoup]]
        List of tuples (url, BeautifulSoup objects of first pages company lists)

    '''

    logger.info(f'Called get_first_pages')

    urls_first_pages: Dict[str, str] = {gen_url_by_industry(ind, 1): ind
                                        for ind in dict_industries}

    results_soup: List[Tuple[str, BeautifulSoup]] = _fetch_chunked_list(urls_first_pages,
                                                                        progress_desc="first pages")

    # switch to industry key
    results_soup_ind: List[Tuple[str, BeautifulSoup]] = [
        (urls_first_pages[url], soup)
        for url, soup in results_soup]

    logger.info(f'... done get_first_pages()')

    return results_soup_ind


def get_all_pages(dict_industries: Dict[str, str]) -> Dict[Tuple[str, int], BeautifulSoup]:
    '''
    Returns all pages of all industries.

    First assesses the number of pages from the first pages. Then fetches the
    remaining ones.

    Parameters
    ----------
    dict_industries: Dict[str, str]
        dictionary (industry id => industry name)

    Returns
    -------
    Dict[Tuple[str, int], BeautifulSoup]
        dictionary (industry, page) => BeautifulSoup

    '''

    soup_all: Dict[Tuple[str, int], BeautifulSoup]
    soup_other_pages: Dict[Tuple[str, int], BeautifulSoup]
    soup_first_pages: Dict[Tuple[str, int], BeautifulSoup]
    dict_n_page: Dict[str, int]

    logger.info(f'Called get_all_pages')

    logger.info(f'Getting BeautifulSoup objects of first pages ...')
    soup_first_pages = {(ind, 1): soup for ind, soup in get_first_pages(dict_industries)}
    logger.info(f'... done getting BeautifulSoup objects of first pages.')

    dict_n_page = {ind: get_page_num(html) for (ind, _), html in soup_first_pages.items()}

    # all urls (excluding first pages)
    dict_urls_ind_page: Dict[str, Tuple[str, int]] = {
            gen_url_by_industry(ind, page): (ind, page)
            for ind in dict_industries
            for page in range(2, dict_n_page[ind] + 1)}

    if dict_urls_ind_page:  # only if there are second pages
        list_url_pages: List[str] = list(dict_urls_ind_page)

        logger.info(f'Getting BeautifulSoup objects of other pages ...')
        soup_other_pages = {
                dict_urls_ind_page[url]: soup 
                for url, soup in _fetch_chunked_list(list_url_pages, 
                                                    progress_desc="remaining pages")}
        logger.info(f'... done getting BeautifulSoup objects of other pages.')

    else:
        soup_other_pages = {}

    logger.info(f'Done get_all_pages()')

    soup_all = {**soup_other_pages,
                **soup_first_pages}

    return soup_all


def parse_industry_table(args):
    '''
    Return table of stocks for one page of one industry. Rows correspond to 
    single stocks with current price data and links to the stock's
    `Kursziele` page.

    NOTE: soup is passed as string, not BeautifulSoup. The latter seems to have pickling
    issues.
    '''

    ind_id, page, soup, dict_industries = args[0]

    soup = BeautifulSoup(soup, 'html.parser')
 
    # logger.info(f'Parsing page {page} of {dict_industries[ind_id]}')

    tb = soup.find_all("table")[2]

    try:
        add_df = pd.read_html(str(tb), thousands='.', decimal=',',
                              converters={'Kurszeit': str}, flavor="bs4")[0]

        add_df['their_date'] = add_df.Kurszeit#pd.to_datetime(add_df.Kurszeit, format='%d%m%Y')
    except AttributeError as e:
        print(e)
        print(f"Raw table: {tb}")
        print(f"Page: {page}")
        raise(e)

    # print(f'{multiprocessing.current_process().name} {page} {ind_id} {len(add_df)}')
    add_df = add_df.assign(page=page, ind_id=ind_id,
                           ind=dict_industries[ind_id])

    add_df['link_0'] = add_df.Name.replace(
            {link.text.replace('  ', ' ').strip(' '):
                link['href'] for link in tb.find_all('a')})

    add_df['price_change'] = add_df['%'].apply(
            lambda x: x if isinstance(x, float)
                        else 0.01 * float(x.strip('%').replace(',', '.')))

    add_df = add_df.rename(columns={'Name': 'name', 'WKN': 'id_wkn',
                                    'Bid': 'bid', 'Ask': 'ask',
                                    'Kurs': 'price'})

    return add_df[['name', 'id_wkn', 'bid', 'ask', 'price', 'price_change',
                   'their_date', 'page', 'ind_id', 'ind', 'link_0']]


def get_companies_table(dict_industries: Dict[str, str]) -> pd.DataFrame:
    """
    """

    if not dict_industries:
        return pd.DataFrame()

    df: pd.DataFrame

    # dictionary (industry, page) => Beautifulsoup
    all_pages: Dict[Tuple[str, int], BeautifulSoup] = get_all_pages(dict_industries)

    logger.info('Fetching full stock table ...')
    list_ind_table = [(ind_id, page, str(soup), dict_industries)
                      for (ind_id, page), soup in all_pages.items()]
    df = parallelize_df(list_ind_table, parse_industry_table)
    logger.info(f"Fetching full stock table done; length={len(df)}.")
 
    df["datetime"] = pd.to_datetime(datetime.utcnow())
    df["their_datetime"] = parse_date(df.their_date)

    # clean duplicate occurrences due to stocks being included in multiple ind
    df_full = (df.groupby([c for c in df.columns
                           if not c in ['ind', 'ind_id', 'page']]).first()
                         ).reset_index()

    return df_full



class DetailTable(metaclass=abc.ABCMeta):
    """
    Abstract base class for target and trend tables
    """

    _cols_required: Set[Any]
    _table: Optional[pd.DataFrame] = None

    def check_cols(required_cols: Set[Any]):
        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):

            df: pd.DataFrame = args[0]
            assert required_cols.issubset(colset := set(df.columns)), \
                f"Missing columns: {required_cols - colset}"

            return wrapped(*args, **kwargs)
        return wrapper


    @property
    def table(self):
        if not self._table is None:
            return self._table
        else:
            raise AttributeError("Table ist not yet defined.")


    @property
    @abc.abstractmethod
    def cols_required(self):
        return self._cols_required

    @property
    @abc.abstractmethod
    def log_name(self):
        return self._log_name


    @table.setter
    def table(self, value):
        @DetailTable.check_cols(self.cols_required)
        def get_checked_table(value):
            return value

        self._table = get_checked_table(value)


    @abc.abstractmethod
    def _expand_url(self, url_0: str) -> str:
        pass

    @abc.abstractmethod
    def _format(self):
        pass

    @check_cols({"link_0", "id_wkn", "name", "ind"})
    def __init__(self, df):
    
        self.df = df
        df["link"] = df.link_0.apply(self._expand_url)
        self.dict_url: Dict[str, str] = df.set_index('link').id_wkn.to_dict()
        self.list_urls: List[str] = list(self.dict_url)


    def make_table(self) -> __qualname__:
        """
        Retrieves all price target pages for each company defined in self.list_urls.

        Returns
        -------
        self

        """

        containing_pages_html: List[Tuple[str, str]]
        df_full_table: pd.DataFrame

        logger.info(f"Getting {self.log_name} table pages ...")
        containing_pages_html = _fetch_chunked_list(self.list_urls, 
                                    as_string=True,
                                    progress_desc=f"Retrieving {self.log_name} pages")
        logger.info(f"... done getting {self.log_name} table pages.")
        logger.info(f"Parsing {self.log_name} table from pages ...")
        df_full_table = parallelize_df(containing_pages_html, self.get_table_single_url)
        logger.info(f"... done parsing {self.log_name} table from pages.")

        self.table = df_full_table.join(self.df.set_index("id_wkn")[["name", "ind"]], 
                                        on="id_wkn")[self.cols_required]

        self.table.date = parse_date(self.table.date)

        return self


    def get_table_single_url(self, args: Tuple[Tuple[str]]) -> pd.DataFrame:
        """

        Parameters
        ----------
        args: Tuple[Tuple[str]]
            - url: URL of the page containing the relevant detailed table; used for logging only
            - html: html string of the page containing the relevant detailed table. 
        
        Returns
        -------
        pd.DataFrame

        """

        df: pd.DataFrame
        url: str
        html: str
        
        url, html = args[0]

        procname: str = multiprocessing.current_process().name
        logstr: str = f'get_table_price_target {procname}, {url}'
        logger.debug(logstr)

        try:
            df = self._format(html)
            df["datetime"] = datetime.utcnow()
            df["id_wkn"] = self.dict_url[url]

        except IndexError as e:
            logger.debug(f"Attempt to retrieve {self.log_name} "
                           f"table from {url} caused IndexError {e}")
            return pd.DataFrame()


        except Exception as e:
            logger.error(f"Unexpected error while parsing table from {url}:")
            logger.error(traceback.format_exc())
            raise e

        logger.debug(f'{self.log_name} table {url}: success')

        return df[self._cols_required]


class PriceTargetTable(DetailTable):

    _cols_required: Set[str] = {"id_wkn", "datetime", "analyst", "dev_price_target", 
                               "price_target", "currency", "date"}
    _log_name: str = "price targets"

    @property
    def cols_required(self):
        return self._cols_required

    @property
    def log_name(self):
        return self._log_name

    def _expand_url(self, url_0: str) -> str:
        """
        Modifies URLs of companies to obtain URL to page with price targets.

        Parameters
        ----------
        url_0: str
            base url to be modified, e.g. "/aktien/kuka-aktie"
        
        Returns
        -------
        str
            modified URL, e.g. '/kursziele/kuka'
        """

        return (f"{BASE_URL}{url_0}"
                        .replace('aktien', 'kursziele')
                        .replace('-aktie', ''))

    def _format(self, html: str) -> pd.DataFrame:
        """
        Class-specific parsing and formatting of DataFrame from input html string.
        Error handling occurs in the calling method `get_table_single_url`.

        Parameters
        ----------
        html: str
            html string of the page containing the table of interest

        Returns
        -------
        pd.DataFrame


        """

        df: pd.DataFrame

        df = [df for df in pd.read_html(html, flavor="bs4") 
              if "Abstand Kursziel" in df.columns][0]
        df = df.rename(columns=lambda x: x.replace('*', ''))

        # filter
        df = df.loc[df['Abstand Kursziel'].str.match('.*\d.*')
                & df['Abstand Kursziel'].str.contains('%')]

        if df.empty:
            return pd.DataFrame(columns=self.cols_required - {"id_wkn", "datetime"})


        # rename and select
        df = df.rename(columns={"Abstand Kursziel": "dev_price_target",
                                "Analyst": "analyst", "Datum": "date",
                                "Kursziel": "price_target"})
        df = df[['analyst', 'dev_price_target', 'price_target', 'date']]
        
        currencies: str = '(CHF|DKK|EUR|\$|€|NOK|SEK|£)'
        newcols: List[str] = ["price_target", "currency"]
        df.loc[:, newcols] = (df.price_target
                                .str.split(currencies, expand=True)
                                .iloc[:, :2].values)
        df.dev_price_target = (df.dev_price_target.astype("string")
                                    .str.replace("(%|\.)", "", regex=True)
                                    .str.replace(",", ".", regex=False)
                                    .astype(float) / 100)
        df.price_target = (df.price_target
                                .str.replace(".", "", regex=False)
                                .str.replace(",", ".", regex=False)
                                .astype(float))

        return df[self.cols_required - {"id_wkn", "datetime"}]



class TrendTable(DetailTable):

    _cols_required: Set[str] = {"id_wkn", "datetime", "analyst", "cat1", "cat2", "date"}
    _log_name: str = "trends"


    @property
    def cols_required(self):
        return self._cols_required


    @property
    def log_name(self):
        return self._log_name


    def _expand_url(self, path_0: str) -> str:
        """
        Modifies path to obtain path with price targets.

        Parameters
        ----------
        path_0: str
            base path to be modified, e.g. "/aktien/kuka-aktie"
        
        Returns
        -------
        str
            modified path, e.g. '/kursziele/kuka'
        """

        return (f"{BASE_URL}{path_0}"
                        .replace('aktien', 'analysen')
                        .replace('-aktie', '-analysen'))

    def _format(self, html: str) -> pd.DataFrame:
        """
        Class-specific parsing and formatting of DataFrame from input html string.
        Error handling occurs in the calling method `get_table_single_url`.

        Parameters
        ----------
        html: str
            html string of the page containing the table of interest

        Returns
        -------
        pd.DataFrame

        """

        df: pd.DataFrame

        soup = BeautifulSoup(html, 'lxml')
        parsed_table = soup.find_all('table')[2] 

        data = [[(div.get_attribute_list("class")[0])
                if (div := td.find("div")) 
                else ''.join(td.stripped_strings)
                for td in row.find_all('td')]
                for row in parsed_table.find_all('tr')]

        df = pd.DataFrame([row for row in data if len(row) == 4],
                        columns=["date", "cat1_0", "analyst", "cat2"])

        if df.empty:
            return pd.DataFrame(columns=self.cols_required - {"id_wkn", "datetime"})

        try:
            name: str = "".join(soup.find("h2").stripped_strings).replace(" Aktie", "")
            name_sanitized: str = name.replace("(", "\(").replace(")", "\)").replace("+", "\+")
            df[["name", "cat1"]] = (df["cat1_0"].str.extractall(f'({name_sanitized}) ?(:?.*)?')
                                            .rename(columns=dict(enumerate(["name", "cat1"])))
                                            .reset_index(-1, drop=True).fillna("none"))
        except Exception as e:
            print(e)
            raise e

        df_missing_cat1 = df.loc[df.cat1.isna()]
        assert df_missing_cat1.empty, f"Found unknown cat1: {df_missing_cat1.cat1_0.unique()}"

        df = df.loc[df.cat2 != ""]
        df.cat2 = df.cat2.replace({"arrow-right": 0,
                                   "arrow-right-top": 1, 
                                   "arrow-right-bottom": -1})

        return df[self.cols_required - {"id_wkn", "datetime"}]


# %%

if __name__ == '__main__':

    logger.setLevel("INFO")

    industry = _CommandLine().industry  # suggestion: 94

    dict_industries = get_industries(drop_zero=True)
    if industry: 
        dict_industries = {key: val for key, val
                           in list(dict_industries.items()) 
                           if key == str(industry)}

    df_companies = get_companies_table(dict_industries)

    table_targets = PriceTargetTable(df_companies).make_table()
    table_trends = TrendTable(df_companies).make_table()

    # df_companies.to_pickle(os.path.join(os.path.normpath("data"), "df_companies.pickle"))
    # table_trends.table.to_pickle(os.path.join(os.path.normpath("data"), "df_trends.pickle"))
    # table_targets.table.to_pickle(os.path.join(os.path.normpath("data"), "df_targets.pickle"))

    engine = create_engine(f"postgresql://{POSTGRES_USERNAME}:{POSTGRES_PASSWORD}@{POSTGRES_IP}:5432/maindb")

    table_targets.table.to_sql("targets", engine, if_exists="replace")
    table_trends.table.query("cat2 != ''").to_sql("trends", engine, if_exists="replace")
    df_companies.to_sql("companies", engine, if_exists="replace")

