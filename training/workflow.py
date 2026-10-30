"""Download -> immutable split export -> detector/classifier training."""
import argparse
from datetime import datetime
import subprocess
import sys
from training.common import DATA
from training.prepare import prepare


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--smoke',action='store_true');p.add_argument('--skip-download',action='store_true')
    a=p.parse_args()
    if not a.skip_download:subprocess.run([sys.executable,'-m','training.fetch_dataset'],check=True)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
    output=DATA/'exports'/stamp
    prepare(output)
    command=[sys.executable,'-m','training.train','--prepared',str(output),'--name',stamp,'--epochs',str(a.epochs)]
    if a.smoke:command+=['--smoke']
    subprocess.run(command,check=True)


if __name__=='__main__':main()
