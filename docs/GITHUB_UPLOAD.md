# GitHub Upload Instructions

The prepared repository is intended for:

```text
https://github.com/AmirNetwork/rmrnet-road-restoration
```

## Option A: Upload Through GitHub Web UI

Use the browser where you are logged in as `AmirNetwork`.

1. Go to `https://github.com/new`.
2. Repository name: `rmrnet-road-restoration`.
3. Description: `Road image restoration for YOLO defect detection and active-contour boundary analysis`.
4. Visibility: public or private.
5. Do not initialize with README, license, or gitignore because they already exist in this folder.
6. Create repository.
7. On the empty repository page, choose uploading an existing file.
8. Upload the contents of this folder, not the parent folder:

```text
github_release/rmrnet-road-restoration/
```

## Option B: Upload With Git Command Line

Install Git for Windows if `git --version` is not available.

Then run:

```powershell
cd C:\Users\61411\Documents\Codex\2026-05-24\i-am-developing-a-new-research\github_release\rmrnet-road-restoration
git init
git branch -M main
git add .
git commit -m "Initial RMR-Net road restoration release"
git remote add origin https://github.com/AmirNetwork/rmrnet-road-restoration.git
git push -u origin main
```

## Option C: Let Codex Upload Later

Create the empty repository under `AmirNetwork`, then connect/install the GitHub app for that account in Codex. After that, Codex can use the GitHub connector to create files and commits.

Current connector caveat from this development session:

```text
Connected GitHub account: amirchoice
Requested repository owner: AmirNetwork
```

The connector cannot create or write to `AmirNetwork/rmrnet-road-restoration` until `AmirNetwork` is connected.

