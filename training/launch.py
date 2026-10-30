"""Open an existing local annotator, or start it hidden and wait until ready."""
import http.client
import json
import os
import subprocess
import sys
import time
import webbrowser

from training.common import DATA, ROOT


def check():
    connection=http.client.HTTPConnection('127.0.0.1',8767,timeout=2)
    try:
        connection.request('GET','/api/health')
        response=connection.getresponse()
        if response.status!=200:return 'occupied'
        data=json.loads(response.read(2048))
        return data if data.get('app')=='dice-local-annotator' else 'occupied'
    except (ConnectionError,TimeoutError,OSError):return None
    except (ValueError,http.client.HTTPException):return 'occupied'
    finally:connection.close()


def main():
    state=check()
    if state=='occupied':raise SystemExit('8767端口被其他程序占用，请先检查。没有停止任何现有服务。')
    if state is None:
        DATA.mkdir(parents=True,exist_ok=True)
        with (DATA/'annotator.stdout.log').open('ab') as out,(DATA/'annotator.stderr.log').open('ab') as err:
            process=subprocess.Popen([sys.executable,'-m','training.annotate'],cwd=ROOT,stdout=out,stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        (DATA/'annotator.pid').write_text(str(process.pid))
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            if process.poll() is not None:raise SystemExit('标注台未能启动，请查看 .runtime/dice-training/annotator.stderr.log')
            if isinstance(check(),dict):break
            time.sleep(.2)
        else:raise SystemExit('启动超时，请检查标注台日志。')
    webbrowser.open('http://127.0.0.1:8767/')
    print('标注台已打开：http://127.0.0.1:8767/；关闭浏览器不会删除标注。')


if __name__=='__main__':main()
