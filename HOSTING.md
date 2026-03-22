# Running Bracket Tracker on Your Mac

This guide walks you through running the Bracket Tracker app locally on your Mac.
The app runs entirely on your computer — no accounts, no cloud servers.

**Why local instead of a hosted URL?**
CBS Sports and ESPN sometimes block requests from cloud server IPs. Running
locally uses your own internet connection, which works reliably every time.

> Currently macOS only. Windows/Linux support coming later.

---

## Requirements

- A Mac running macOS 12 (Monterey) or later
- An internet connection (to download setup tools and scrape bracket data)
- ~500 MB of free disk space (for Python, packages, and Chromium)

---

## Step 1 — Download the app

If you received this as a zip file, unzip it anywhere (e.g. your Desktop or
Documents folder). If you cloned from GitHub, you already have it.

You should have a folder that looks like this:
```
bracket-tracker/
├── run_mac.sh        ← the launcher (start here)
├── app.py
├── requirements.txt
└── ...
```

---

## Step 2 — Run it

Open **Terminal** (search for it in Spotlight with ⌘+Space → type "Terminal").

Then run:
```bash
cd ~/Desktop/bracket-tracker   # adjust path to wherever you put the folder
bash run_mac.sh
```

**The first time you run this**, the script will automatically:
1. Install Homebrew (Mac's package manager) if you don't have it
2. Install Python 3.11+ if needed
3. Create an isolated Python environment for the app
4. Download all required packages
5. Download Chromium (~150 MB, one-time only)

This first-time setup takes **3–5 minutes**. After that, launching the app
takes only a few seconds.

---

## Step 3 — Use the app

Once setup is complete, your browser will open automatically to:
```
http://localhost:5000
```

Fill in the form with your CBS Sports or ESPN credentials and pool info, then
click **Analyze**. Results appear in about 30–60 seconds depending on pool size.

When you're done, go back to Terminal and press **Ctrl+C** to stop the app.

---

## Running it again later

Just run the same command again:
```bash
bash run_mac.sh
```

The app starts in a few seconds — no reinstall, no waiting.

---

## Sharing with family members

Each person runs the app on their own Mac:

1. Send them the `bracket-tracker` folder (zip it and share via AirDrop, email,
   iCloud, Google Drive, etc.)
2. They open Terminal, `cd` to the folder, and run `bash run_mac.sh`
3. Their browser opens and they use it with their own credentials

Their credentials never leave their computer.

---

## Troubleshooting

**"Permission denied" when running the script**
```bash
chmod +x run_mac.sh
bash run_mac.sh
```

**Homebrew install asks for your password**
That's normal — Homebrew needs admin access to install to `/usr/local` or
`/opt/homebrew`. Enter your Mac login password.

**"Port 5000 is already in use"**
Something else is using port 5000. The script detects this and opens the
existing app. If that doesn't work, quit the other app using port 5000
(check System Settings → AirPlay Receiver uses 5000 on some Macs — you can
disable it there) and run the script again.

**Login fails for CBS or ESPN**
- Double-check your email and password
- Make sure the pool URL is correct (copy it from the pool Standings page)
- Try logging in to CBS/ESPN in your regular browser first to confirm credentials work

**Chromium won't launch / browser crashes**
```bash
# Reinstall Chromium
source .venv/bin/activate
playwright install chromium
```

**The app is slow on first analyze**
Scraping 10–20 brackets takes 30–60 seconds — that's normal. Larger pools
(20+ entries) may take up to 2 minutes.

---

## Updating the app

When a new version is available, replace the folder contents with the new files
and run `bash run_mac.sh` — it will update packages automatically.

---

## Privacy

- Your CBS/ESPN credentials are entered in the form and used only for that
  session. They are not saved to disk.
- All scraping happens on your machine via your internet connection.
- No data is sent to any external server (other than CBS/ESPN to fetch brackets).
