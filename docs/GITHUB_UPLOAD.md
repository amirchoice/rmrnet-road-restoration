# GitHub Upload / Transfer Instructions

The repository was published to:

```text
https://github.com/amirchoice/rmrnet-road-restoration
```

## If You Want To Move It To AmirNetwork

The original requested owner was `AmirNetwork`, but the connected and signed-in GitHub account during publication was `amirchoice`. To move the repository later:

1. Open the repository on GitHub.
2. Go to `Settings`.
3. Scroll to `Danger Zone`.
4. Use `Transfer ownership`.
5. Enter `AmirNetwork` as the new owner if that account or organization is available to you.

## Re-Push From Local Git

```text
C:\Users\61411\Documents\Codex\2026-05-24\i-am-developing-a-new-research\github_release\rmrnet-road-restoration
```

The remote currently points to:

```powershell
https://github.com/amirchoice/rmrnet-road-restoration.git
```

To push to a different owner later:

```powershell
git remote set-url origin https://github.com/AmirNetwork/rmrnet-road-restoration.git
git push -u origin main
```

Only do this after creating an empty repository or transferring ownership.

