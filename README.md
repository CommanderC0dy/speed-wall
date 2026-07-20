# Speed Wall — hitrostna lestvica za plezalni center

One static page (`index.html`) that shows verified fastest times, with a ▶ video
button that plays each climber's proof video right on the page.

Flow: climb → film → submit Google Form → organizer verifies the video →
leaderboard updates automatically → QR code on the wall shows live standings.

## Live pieces (already wired into index.html)

- Form (submissions): https://docs.google.com/forms/d/1z6sPQHyzv2hEeM7lncZbpxmWOyNZKaVgK7IvvCw5WUs/viewform
- Response sheet: "Hitrostno" — columns `Timestamp, Ime, Cas, Video`
- Published CSV feed: `.../pub?output=csv` (set as SHEET_CSV_URL)
- Video folder: "Untitled form (File responses)" in Drive

## Remaining setup

1. **Share the video folder** — in Drive, right-click
   "Untitled form (File responses)" → Share → **Anyone with the link: Viewer**.
   Without this, only the organizer can watch the videos.
2. **Add the Verified column** — in the "Hitrostno" sheet, type `Verified` in
   cell **E1**. To approve a submission, watch its video and type `x` (or
   anything) in that row's Verified cell. Empty = hidden from the leaderboard.
3. **Host it** — GitHub Pages (free): create a repo, upload `index.html`,
   Settings → Pages → deploy from `main`.
4. **QR code** — point it at the GitHub Pages URL.

## Rules encoded in the page

- Only rows with something in the **Verified** column are shown.
- Best time per climber is kept; times accept `12.34`, `12,34` or `1:23.4`.
- Data refreshes every minute (published CSV itself lags edits by a minute or
  two — that's Google's cache, not a bug).
- Don't rename the sheet's header row — the page finds columns by the words
  `ime`, `cas`, `video`, `verified` (case-insensitive). If you add more
  boulders later, add a form question whose title contains `balvan` and the
  page grows boulder tabs automatically.

## Notes

- Form uploads require climbers to sign in with a Google account.
- Videos count against the organizer's 15 GB free Drive storage — tell people
  to trim clips to just the attempt.
