#!/usr/bin/env python3
"""
Resume supervisor for the IBKR 5-min pull.
TWS forces a daily re-login that drops the API connection. This wrapper makes the
download survive that bounce WITHOUT intervention:
  - waits for the TWS port (7496) to be reachable,
  - launches 2__IntradayHistoricalDownloader.py (resumable: skips finished names),
  - if TWS drops mid-run, kills the child and waits for re-login, then relaunches,
  - stops once all names are present, or after 3 no-progress runs (= remaining names
    are un-gettable), or a wall-clock safety cap.
Resume is lossless: files are written per-ticker on completion; a half-written or
killed-mid-ticker file fails to read and is simply re-fetched next pass.
"""
import subprocess, time, glob, socket, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

CMD = [sys.executable, '-u', '2__IntradayHistoricalDownloader.py',
       '--universe', 'traded', '--bar-size', '5min', '--lookback-days', '730',
       '--num-threads', '1', '--ignore-market-hours']
TARGET   = 805
MAXWALL  = 14 * 3600
PORT     = 7496

def n_done():
    return len(glob.glob('Data/IntradayData/*_5min.parquet'))

def tws_up():
    s = socket.socket(); s.settimeout(2.5)
    try:
        s.connect(('127.0.0.1', PORT)); return True
    except Exception:
        return False
    finally:
        try: s.close()
        except Exception: pass

def log(m):
    print(f'[sup {time.strftime("%H:%M:%S")}] {m}', flush=True)

START = time.time(); plateau = 0; attempt = 0
log(f'supervisor start | {n_done()}/{TARGET} files present | TWS {"up" if tws_up() else "down"}')

while time.time() - START < MAXWALL:
    if n_done() >= TARGET:
        log('all names present; done'); break
    if plateau >= 3:
        log(f'3 no-progress runs -> remaining names un-gettable; stopping at {n_done()}/{TARGET}'); break

    if not tws_up():
        log('TWS down; waiting for re-login...')
        w = 0
        while not tws_up() and time.time() - START < MAXWALL:
            time.sleep(15); w += 15
            if w % 120 == 0:
                log(f'  ...still waiting for TWS ({w}s elapsed), {n_done()}/{TARGET} done')
        if tws_up():
            log('TWS back up'); time.sleep(5)
        else:
            break

    before = n_done(); attempt += 1
    log(f'attempt {attempt}: launching downloader ({before}/{TARGET} done)')
    try:
        p = subprocess.Popen(CMD)
    except Exception as e:
        log(f'launch failed: {e}'); time.sleep(30); continue

    tws_dropped = False
    while p.poll() is None:
        time.sleep(15)
        if not tws_up():
            log('TWS dropped mid-run -> terminating downloader to relaunch cleanly')
            tws_dropped = True
            try: p.terminate()
            except Exception: pass
            try: p.wait(timeout=30)
            except Exception:
                try: p.kill()
                except Exception: pass
            break

    after = n_done(); gained = after - before
    log(f'attempt {attempt} ended (rc={p.returncode}) | gained {gained} | now {after}/{TARGET}')
    if not tws_dropped and tws_up() and gained == 0:
        plateau += 1; log(f'no progress with TWS up (plateau {plateau}/3)')
    else:
        plateau = 0
    time.sleep(8)

log(f'supervisor exit | {n_done()}/{TARGET} files | {(time.time()-START)/3600:.2f}h elapsed')
