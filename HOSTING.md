# Hosting Setup Guide

This guide walks you through putting the Bracket Tracker web app online so
family members can use it at a sharable URL — no install required on their end.

We'll use **Render** (render.com). It's free for this use case and has a
one-click deploy from GitHub. The whole setup takes about 10 minutes.

---

## Prerequisites

You need two things before starting:

1. **This repo on GitHub.** If it's already there, skip ahead.
   If not — go to [github.com](https://github.com), create a new repository,
   then push this folder to it:
   ```
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

2. **A free Render account.** Sign up at [render.com](https://render.com) —
   you can use your GitHub account to log in, which makes the next steps easier.

---

## Step 1 — Connect your GitHub repo to Render

1. Log in to [render.com](https://render.com)
2. Click **New +** in the top-right corner
3. Choose **Web Service**
4. Click **Connect account** next to GitHub (if you haven't already)
   - Authorize Render to access your repositories
5. Find your bracket tracker repo in the list and click **Connect**

---

## Step 2 — Configure the service

Render will read `render.yaml` automatically and pre-fill most settings.
Verify these values look right before deploying:

| Setting | Value |
|---|---|
| **Name** | `bracket-tracker` (or anything you like) |
| **Runtime** | Python |
| **Build Command** | `pip install -r requirements.txt && playwright install chromium --with-deps` |
| **Start Command** | `gunicorn app:app --timeout 120 --workers 2` |
| **Instance Type** | Free |

> The build command installs Chromium (the headless browser used for scraping).
> This is the step that takes the longest the first time (~3-4 minutes).

---

## Step 3 — Set the SECRET_KEY environment variable

Render should auto-generate this from `render.yaml`, but double-check:

1. Scroll down to the **Environment Variables** section
2. Confirm there's a `SECRET_KEY` entry with a value like `Generate`
3. If it's missing, click **Add Environment Variable**:
   - Key: `SECRET_KEY`
   - Value: click **Generate** (Render will create a random secure value)

You don't need to set CBS or ESPN credentials here — users enter those
themselves in the web form each time they use the app.

---

## Step 4 — Deploy

Click **Create Web Service** at the bottom of the page.

Render will:
1. Pull your code from GitHub
2. Run the build command (installs Python packages + Chromium)
3. Start the app with gunicorn

Watch the deploy log scroll by. When you see a line like:
```
==> Your service is live 🎉
```
…you're done. Render will show you a URL at the top of the page that looks like:
```
https://bracket-tracker.onrender.com
```

Copy that URL — that's what you share with family.

---

## Step 5 — Test it

Open the URL in your browser. You should see the Bracket Tracker form.
Run through it with your own CBS or ESPN credentials to make sure everything works.

If something goes wrong, click **Logs** in the Render dashboard to see the error output.

---

## Sharing with family

Send them the Render URL. That's it. They:

1. Open the link in any browser (phone or computer)
2. Choose CBS Sports or ESPN
3. Enter their credentials + pool URL + bracket name
4. See their results

They don't need to install anything.

---

## Important: the free tier "sleeps"

Render's free tier spins down the app after 15 minutes of no traffic.
The **first person to visit after a sleep period waits about 30–60 extra seconds**
for the app to wake up — then it's fast.

If that's annoying, you can upgrade to the **Starter** plan ($7/month) to keep
it always-on. During March Madness you might get a few complaints about the
slow first load, so it's worth it for a few weeks.

---

## Updating the app later

Any time you push new code to GitHub, Render will automatically redeploy.
You don't need to do anything in the Render dashboard.

```
# Make your changes, then:
git add .
git commit -m "describe your change"
git push
```

Render picks it up within a minute or two.

---

## Troubleshooting

**The deploy failed during build**
- Check the build log for the specific error
- Most common cause: a Python package version conflict. Try removing the
  version pins from `requirements.txt` and redeploy.

**Login fails for CBS/ESPN on the hosted app but works locally**
- CBS and ESPN occasionally block requests from cloud server IP ranges
- For ESPN: see the "cookie mode" note in the code comments (use `espn_s2` +
  `SWID` cookies instead of email/password)
- For CBS: try again later — CBS rate-limits scraping from datacenter IPs

**The app times out**
- Large pools (20+ entries) can take 60-90 seconds to scrape
- Go to your Render service settings → **Instance Type** → upgrade to at least
  **Starter** and increase the timeout in the start command to `--timeout 180`

**I need to change the URL / custom domain**
- In the Render dashboard, go to your service → **Settings** → **Custom Domain**
- Add a domain you own (e.g. `brackets.yourdomain.com`) and follow the DNS instructions
