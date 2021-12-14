from apscheduler.schedulers.blocking import BlockingScheduler
import datetime


sched = BlockingScheduler()

# @sched.scheduled_job('cron', day_of_week='mon-fri', hour=5)
@sched.scheduled_job('interval', seconds=5)
def timed_job():
    print('This job is run regularly.')
    with open("cron_test.dat", "a") as f:
        f.writelines(["Test", str(datetime.datetime.now()), "\n"])

sched.start()

