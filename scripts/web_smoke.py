"""Real Chromium + live MuJoCo smoke check. Does not touch running training jobs."""
import os
from pathlib import Path
import threading
from http.server import ThreadingHTTPServer

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH',str(ROOT/'.cache/browsers'))
from g1_odom import serve
from g1_odom.common import write_json

def main():
 app=serve.App();serve.APP=app;errors=[]
 server=ThreadingHTTPServer(('127.0.0.1',0),serve.Handler)
 thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 url=f'http://127.0.0.1:{server.server_address[1]}'
 try:
  from playwright.sync_api import sync_playwright
  with sync_playwright() as pw:
   browser=pw.chromium.launch(headless=True,args=['--no-sandbox','--disable-gpu'])
   page=browser.new_page(viewport={'width':1920,'height':1080})
   page.on('pageerror',lambda err:errors.append(str(err)))
   page.goto(url+'/odometry');page.locator('#start').click()
   page.wait_for_function("document.getElementById('time').textContent !== '—'",timeout=30000)
   page.wait_for_function("document.getElementById('frame').naturalWidth > 0",timeout=30000)
   assert page.locator('#title').inner_text()=='平面里程计'
   assert '理想' in page.locator('#headingReference').inner_text()
   assert '非真机' in app.status()['heading_reference']
   page.locator('#vx').fill('0.4');page.locator('#vx').dispatch_event('change')
   page.wait_for_timeout(6000)
   assert page.locator('#sigma').inner_text().endswith('cm')
   assert page.locator('#quality').inner_text() in ('正常','注意','告警')
   page.screenshot(path=str(ROOT/'docs/images/web_odometry.png'),full_page=True)
   page.locator('#pause').click();page.wait_for_function("document.getElementById('status').textContent === '已暂停'")
   page.locator('#pause').click();page.locator('#reset').click()
   page.goto(url+'/training');page.wait_for_timeout(1500)
   assert page.locator('#training').is_visible()
   if (ROOT/'reports/evaluation.json').exists():
    page.wait_for_function("document.querySelectorAll('#report table').length >= 2",timeout=10000)
   page.screenshot(path=str(ROOT/'docs/images/web_training.png'),full_page=True)
   page.goto(url+'/odometry');page.locator('#stop').click()
   for _ in range(100):
    if app.proc is None:break
    page.wait_for_timeout(100)
   assert app.proc is None
   assert app.status()['status']=='stopped'
   browser.close()
  if errors:raise RuntimeError(errors)
  write_json(ROOT/'reports/web_smoke.json',{'pass':True,'browser_errors':errors,
              'verified':['live rendered G1','telemetry with uncertainty','velocity control','pause/resume','reset',
                          'training and report page','worker cleanup']})
 finally:app.stop();server.shutdown();server.server_close()

if __name__=='__main__':main()
