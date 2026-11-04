# RR# AI ATS Resume Analyzer

Upload a resume (PDF, DOCX or TXT) and get:

- An estimated **ATS score (0-100)** with a category breakdown
- **Prioritized improvement suggestions** and bullet-point rewrite examples
- Optional **job description matching** (match score, matched and missing keywords)
- A **downloadable report**

Built with Streamlit and Google Gemini Flash.

> The score is an AI-generated estimate of how ATS software and recruiters may read a resume. It is not the output of a real ATS.

## Files

```
app.py            the whole app
requirements.txt  pinned dependencies
README.md         this file
```

## 1. Get a Gemini API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key** and copy it.

## 2. Run locally

Requires Python 3.10 or newer.

```bash
# create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
```

Create a file named `.env` next to `app.py`:

```
GEMINI_API_KEY=paste_your_key_here
```

Start the app:

```bash
streamlit run app.py
```

It opens at http://localhost:8501.

Optional: set `GEMINI_MODEL` in `.env` to use a different Gemini Flash model. The default is `gemini-2.5-flash`.

## 3. Put the project on GitHub

First create a `.gitignore` file so your key is never uploaded. Put this inside it:

```
.env
.streamlit/secrets.toml
venv/
__pycache__/
```

Then:

1. Create a free account at https://github.com.
2. Click **New repository**, name it (for example `ats-resume-analyzer`), keep it **Public** (the free Streamlit plan needs this for simplest setup), and do **not** add a README (you already have one). Click **Create repository**.
3. Install Git from https://git-scm.com if you do not have it.
4. In your project folder run:

```bash
git init
git add app.py requirements.txt README.md .gitignore
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ats-resume-analyzer.git
git push -u origin main
```

If Git asks for a password, GitHub needs a Personal Access Token instead: GitHub -> Settings -> Developer settings -> Personal access tokens -> generate one with `repo` access, and paste it as the password.

Check on GitHub that `.env` is **not** listed. If you ever pushed a key by accident, delete that key in Google AI Studio and create a new one.

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app** (or **New app**) and choose **Deploy a public app from GitHub**.
3. Pick your repository, branch `main`, and main file path `app.py`.
4. Click **Advanced settings**:
   - Choose Python 3.11 or 3.12.
   - In the **Secrets** box paste:
     ```toml
     GEMINI_API_KEY = "paste_your_key_here"
     ```
5. Click **Deploy**. The first build takes a couple of minutes.

To change the key later: open your app -> **Manage app** -> **Settings** -> **Secrets**, edit, and save. Pushing new code to GitHub redeploys automatically.

## Troubleshooting

| Problem | Fix |
|---|---|
| "API key missing" in the sidebar | Check `.env` locally, or the Secrets box on Streamlit Cloud. The name must be exactly `GEMINI_API_KEY`. |
| "Your Gemini API key was rejected" | The key is wrong or disabled. Create a new one and remove extra spaces. |
| "Rate limit / quota reached" | The free tier has per-minute and daily limits. Wait and retry. The app already retries automatically with increasing delays. |
| "Gemini is overloaded (503)" | Temporary on Google's side. The app retries up to 4 times; try again in a minute if it still fails. |
| "Model not found" | Model names change over time. Set `GEMINI_MODEL` (locally in `.env`, or as another line in Secrets) to a current Gemini Flash model. |
| "Almost no text could be read" | The file is probably a scanned image. Upload a text-based PDF, DOCX or TXT. |
| Old `.doc` files fail | Save the resume as `.docx` first. |

## Privacy

Uploaded resumes are sent to Google's Gemini API for analysis and are not stored by this app.
