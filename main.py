import datetime
# import datetime
import time
import boto3
import schedule
from botocore.exceptions import ClientError
import camera
import config
from Distance.distance import get_distance


def interaction():
    source = []
    dateObj = datetime.datetime.now()
    dateString = dateObj.strftime("%Y-%m-%d")
    cam1 = f'cam1_{dateString}.mp4'
    cam2 = f'cam2_{dateString}.mp4'
    source.append(cam1)
    source.append(cam2)
    camera.set_source(source)
    get_distance()


def download():
    client_s3 = boto3.client(
        's3',
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.access_secret
    )
    dateObj = datetime.datetime.now()
    dateString = dateObj.strftime("%Y-%m-%d")
    cam1 = f'cam1_{dateString}.mp4'
    cam2 = f'cam2_{dateString}.mp4'
    try:
        client_s3.download_file(config.bucket_name, "Video5.mp4", cam1)
        client_s3.download_file(config.bucket_name, "Video2_Trim.mp4", cam2)
    except ClientError as e:
        print(e)
    except Exception as e:
        print(e)


def main():
    schedule.every().day.at("21:00")
    schedule.every().day.at("22:30").do(interaction)
    while True:
        schedule.run_pending()
        time.sleep(1)


main()
