FROM python:3.10.1-bullseye

RUN apt-get update && apt-get -y dist-upgrade
RUN apt-get install -y bash tree cron 

# RUN apt-get -y install python3-numpy 
# RUN apt-get -y install python3-pandas


RUN mkdir -p /home/jovyan/app
WORKDIR /home/jovyan/app

RUN pip install pandas==1.3.5 

RUN git clone https://github.com/theskumar/python-dotenv.git

COPY ./requirements_backend.txt .
RUN pip3 install -r requirements_backend.txt

RUN pip3 install lxml

ENV PYTHONPATH=$PYTHONPATH:/home/jovyan/app/python-dotenv/src  
#:/usr/lib/python3/dist-packages

COPY ./backend.py .
ADD  ./utils ./utils
COPY ./.env .
COPY ./run_backend.sh .

RUN mkdir ./data


# from https://stackoverflow.com/questions/37458287/how-to-run-a-cron-job-inside-a-docker-container
COPY reccoll-cron /etc/cron.d/
RUN chmod 0644 /etc/cron.d/reccoll-cron
RUN crontab /etc/cron.d/reccoll-cron
RUN touch /var/log/cron.log


CMD ["cron", "-f"]

