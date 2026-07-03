# Insurance Claim Risk Demo (Streamlit)

A small web app that predicts the probability that an active, credit-linked
insurance contract will file a claim, based on a policyholder's profile.
It trains a simple, transparent model from `data/dataset_Sinistre.csv`
every time it starts, so there's no separate model file to manage.

This README assumes **zero prior experience**. Follow it top to bottom.

---

## Part 1 — Run the app on your own computer

### Step 1: Install Python
1. Go to https://www.python.org/downloads/
2. Click the big yellow "Download Python" button (get version 3.10, 3.11, or 3.12).
3. Run the installer.
   - **Windows:** on the first screen, tick the box **"Add python.exe to PATH"** before clicking Install. This step is easy to miss and causes most beginner problems.
   - **Mac:** run the downloaded `.pkg` file and click through the installer.
4. Confirm it worked. Open a terminal:
   - **Windows:** press the Windows key, type `cmd`, press Enter.
   - **Mac:** press Cmd+Space, type `terminal`, press Enter.
5. Type this and press Enter:
   ```
   python --version
   ```
   (On Mac, if that doesn't work, try `python3 --version`.)
   You should see something like `Python 3.11.5`. If you see an error, redo Step 3 and make sure PATH was checked.

### Step 2: Get the project folder onto your computer
You already have this folder (`insurance-risk-demo`) from this conversation.
Unzip it somewhere easy to find, like your Desktop.

### Step 3: Open a terminal *inside* the project folder
This is the step people get stuck on. The terminal needs to be "standing inside"
the folder that contains `app.py`.

- **Windows:** open the `insurance-risk-demo` folder in File Explorer, click on
  the empty address bar at the top, type `cmd`, press Enter. A terminal opens
  already inside that folder.
- **Mac:** open Finder, find the folder, right-click it, choose
  **"New Terminal at Folder"** (if you don't see this option, open Terminal
  normally and type `cd ` followed by dragging the folder into the terminal
  window, then press Enter).

Check you're in the right place:
```
dir        (Windows)
ls         (Mac)
```
You should see `app.py`, `requirements.txt`, `README.md`, and a `data` folder listed.

### Step 4: Create a virtual environment (a clean, isolated Python setup)
This keeps this project's packages separate from everything else on your
computer. Copy-paste each line, pressing Enter after each one.

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac:**
```
python3 -m venv venv
source venv/bin/activate
```

After running the second command, you should see `(venv)` appear at the start
of your terminal line. That means it worked. You'll need to run the
"activate" command again every time you close and reopen the terminal.

### Step 5: Install the required packages
```
pip install -r requirements.txt
```
This downloads Streamlit, pandas, scikit-learn, and a couple of other
libraries. It can take 1–3 minutes. Some yellow warning text is normal;
only worry if you see a line starting with `ERROR`.

### Step 6: Run the app
```
streamlit run app.py
```
Your browser should open automatically at `http://localhost:8501` showing
the app. If it doesn't open automatically, copy that address into your
browser manually.

### Step 7: Stop the app
Click back into the terminal window and press `Ctrl + C`.

### Running it again later
Every time you come back to work on this:
```
cd path/to/insurance-risk-demo
venv\Scripts\activate        (Windows)   OR   source venv/bin/activate   (Mac)
streamlit run app.py
```

---

## Part 2 — Using GitHub as a complete beginner

GitHub is a website that stores a copy of your code online, keeps a history
of every change, and lets you deploy the app for free later. You don't need
to be a programmer to use it — think of it like Google Drive for code, with
a memory of every version.

### Step 1: Create a GitHub account
1. Go to https://github.com
2. Click **Sign up**, follow the prompts (email, password, username).

### Step 2: Install GitHub Desktop (the beginner-friendly way — no typing git commands)
1. Go to https://desktop.github.com and download it for your OS.
2. Install it and sign in with the GitHub account you just made.

### Step 3: Create a new repository ("repo" = project folder on GitHub)
1. Open GitHub Desktop.
2. Go to **File → New Repository**.
3. Fill in:
   - **Name:** `insurance-risk-demo`
   - **Local path:** choose the folder *containing* your `insurance-risk-demo`
     project folder (or point it at the existing folder — GitHub Desktop
     will ask to add existing files).
   - Tick **Initialize this repository with a README** only if the folder
     is currently empty; since you already have files, instead choose
     **"Add an Existing Repository"** if GitHub Desktop detects your folder
     already has files, and select your `insurance-risk-demo` folder.
4. Click **Create Repository** (or **Add Repository**).

### Step 4: Make your first commit (a "commit" = a saved snapshot of your changes)
1. In GitHub Desktop, you'll see a list of all your files under "Changes".
2. In the bottom-left box, write a short summary, e.g. `Initial version of the risk demo`.
3. Click **Commit to main**.

### Step 5: Publish it to GitHub.com
1. Click the **Publish repository** button at the top.
2. Choose whether it's **Public** (anyone can see the code) or **Private**
   (only you). For a school project demo, Public is usually fine unless
   the data is sensitive — if `dataset_Sinistre.csv` contains real personal
   data, choose **Private**.
3. Click **Publish Repository**.

Your code is now on GitHub. You can view it by clicking **Repository → View on GitHub**.

### Step 6: Making future changes
Whenever you edit `app.py` or any file:
1. Open GitHub Desktop — it will show you what changed.
2. Write a short commit message describing the change.
3. Click **Commit to main**.
4. Click **Push origin** (this uploads your commit to GitHub.com).

That's the entire day-to-day workflow: **edit → commit → push**.

### (Optional) If you ever prefer typed commands instead of GitHub Desktop
```
git init
git add .
git commit -m "Initial version of the risk demo"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/insurance-risk-demo.git
git push -u origin main
```
You'd only need this if you're on a computer without GitHub Desktop installed.

---

## Part 3 — Put the demo online for free (Streamlit Community Cloud)

Once your code is on GitHub, you can get a public web link in a few clicks.

1. Go to https://share.streamlit.io and sign in with your GitHub account.
2. Click **Create app** (or **New app**).
3. Choose:
   - **Repository:** `YOUR-USERNAME/insurance-risk-demo`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy**.
5. Wait 1–2 minutes while it installs everything and starts the app.
6. You'll get a public link like `https://your-app-name.streamlit.app` that
   you can share with anyone — no installation needed on their end.

If it fails to deploy, click **Manage app → Logs** to see the error message,
and check that `requirements.txt` and `app.py` are both at the top level of
the repository (not inside a subfolder).

---

## What's in this folder

```
insurance-risk-demo/
├── app.py                        # The Streamlit app (all the logic)
├── requirements.txt              # List of packages needed to run it
├── data/
│   └── dataset_Sinistre.csv      # The claims dataset used to train the model
└── README.md                     # This file
```

## How the model works (short version)

- Trains a **Logistic Regression** model with cost-sensitive class weighting
  (handles the fact that claims are rare, without creating synthetic data).
- Uses the same leakage-safe feature set as the research notebook: excludes
  `Cout_sinistre`, `Prob_sinistre`, `Risk_Score`, and `Risk_Segment`, since
  those are outcome/derived fields that would leak the answer.
- Only models **active contracts**, since claims can only occur on those.
- Log-transforms skewed numeric fields (income, loan amount, outstanding capital).
- Picks a decision threshold that maximizes F1 while keeping recall ≥ 50%,
  matching the actuarial floor used in the notebook.

The app retrains itself automatically each time it starts (usually a few
seconds), so there's no separate model file to keep in sync.
