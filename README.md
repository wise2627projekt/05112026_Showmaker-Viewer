# PhotoVec Show Maker + Viewer (Windows)

Two small programs for making and playing **photo shows**:

* **PhotoVecShowMaker.exe** - you add photos, (optionally) draw a mask, name the photos, set the sliders and press *Render show*. It writes one file ending in `.photovecshow`.
* **PhotoVecViewer.exe** - plays such a file: a grid of photos; click one and move the sliders to go from the photo to a vector drawing of it.

This repository builds both programs on GitHub's own computers, so **nothing needs to be installed on your PC**. The result is a zip file you can send to a friend.

---

## Step by step: build the programs and send them to a friend

### 1. Put the files into a new GitHub repository
1. Go to <https://github.com> and sign in (or create a free account).
2. Click the **+** at the top right → **New repository**.
3. Name it e.g. `photovec-show`. Choose **Private** (only you can see it) or Public. Leave "Add a README" **unticked**. Click **Create repository**.
4. On the empty repository page click **uploading an existing file** (the blue link in the middle of the page).
5. Unzip the zip you got and open the folder `photovec-show`. Select **everything inside it** (the folders `photovec` and `.github`, and the files `PhotoVecShow.spec`, `photovec_show_maker.py`, `photovec_viewer.py`, `requirements.txt`, `requirements-flex.txt`, `README.md`, `.gitignore`) and **drag it all into the browser window**. Wait until every file shows as uploaded (about 70 files).
6. Scroll down and click **Commit changes**.

> If you cannot see the folder `.github` in your file explorer, enable *View → Show → Hidden items* first. Or skip it and do this instead after step 6: **Add file → Create new file**, type the name `.github/workflows/build.yml` (typing the slashes creates the folders), paste the contents of the file `build.yml` from that folder, and click **Commit changes**.

### 2. Let GitHub build the programs
1. Click the **Actions** tab of your repository. (If GitHub asks you to enable workflows, click the green button.)
2. In the list on the left click **Build Show Maker and Viewer**.
3. Click **Run workflow** (right side) → the green **Run workflow** button.
4. Wait about **10-15 minutes**. A yellow dot means it is running; a green check means it is done. Click the run to see it.

### 3. Download the result
1. Click the finished (green) run.
2. At the bottom of the page, under **Artifacts**, click **PhotoVec-ShowMaker-Viewer-Windows**. Your browser downloads a zip file. (It is big, a few hundred MB, because it contains all libraries.)
3. Send that zip to your friend with a file-sharing link (e-mail is usually too small).

The download stays on GitHub for 90 days; build again any time with step 2.

### 4. What your friend does
1. **Unzip the whole zip** to a folder (right click → *Extract All*). Do not run the programs from inside the zip.
2. Keep both `.exe` files in that folder, **together with the `_internal` folder** next to them.
3. Double-click **PhotoVecViewer.exe**, then open a `.photovecshow` file you sent along (or drag the file onto the window).
4. Windows may say *"Windows protected your PC"* because the programs are not signed: click **More info → Run anyway**. Some antivirus programs also warn about unknown programs; the programs only read the files you open.

---

## Making a show (Show Maker)
1. **Add photos…** (a few photos, or many).
2. For every photo: type a **name** (shown on the picture like a film subtitle), choose the *Threshold range*, optionally **Draw mask…** so only a part is traced, and leave the sliders where the photo should start.
3. Under **Name style (all photos)** you can change the font, bold/italic, size, colours and outline of the names, and let the name fade in with the first slider.
4. **Render show…** and choose where to save the `.photovecshow` file. Send that file to your friend; it contains everything.

Fonts: the names use *Inter* (free, included). A font you choose by name must also be installed on the other computer, or tick *Put the font file into the show* when you load a font file whose licence allows it.

## If something goes wrong
* **No "Run workflow" button / workflow not listed:** the file `.github/workflows/build.yml` is missing or in the wrong place (see the note in step 1).
* **Red X on the run:** click the run → click the step with the red X → copy the last 20 lines of its log and send them to whoever helped you set this up.
* **GitHub shows a warning about "Node.js 20":** harmless, the build still works.

## Contents
`photovec/` (the program's Python files and fonts), `photovec_show_maker.py` / `photovec_viewer.py` (start files), `PhotoVecShow.spec` (build recipe), `requirements*.txt` (libraries), `.github/workflows/build.yml` (the build on GitHub). Fonts in `photovec/fonts` have their own free licences (SIL Open Font License; see the `*-OFL.txt` files there).
