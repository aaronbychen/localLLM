import os

for k, v in os.environ.items():
    print(f'{k}={v}')


# home_dir =os.environ['HOME']
DB_PWD = os.environ['DB_PWD']
print(DB_PWD)