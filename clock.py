from apscheduler.schedulers.blocking import BlockingScheduler
import datetime


sched = BlockingScheduler()

@sched.scheduled_job('cron', day_of_week='mon-fri', hour=15)
#@sched.scheduled_job('interval', seconds=5)
def schedule1_job():
    print('This job is run at hour=15.')
    with open("cron_test.dat", "a") as f:
        f.writelines(["Test", str(datetime.datetime.now()), "\n"])

@sched.scheduled_job('cron', day_of_week='mon-fri', hour=14, minute=30)
def schedule2_job():
    print('This job is at hour=14, minute=30.')
    with open("cron_test.dat", "a") as f:
        f.writelines(["Test", str(datetime.datetime.now()), "\n"])

sched.start()

