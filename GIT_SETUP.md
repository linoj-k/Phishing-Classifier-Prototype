# Git Setup Guide for Your Phishing Classifier

## Quick Start (Copy-Paste These Commands)

Open your terminal in the `Classifier Prototype` folder and run:

```bash
# 1. Initialize Git repository
git init

# 2. Add all your code files
git add .

# 3. Create your first commit
git commit -m "Initial commit: Phishing URL classifier with Random Forest"

# 4. Check status
git status
```

✅ Done! Your project is now under version control.

---

## What Just Happened?

- **git init**: Created a hidden `.git` folder to track changes
- **git add .**: Staged all your files (except those in `.gitignore`)
- **git commit**: Saved a snapshot of your code
- **git status**: Shows what's changed

---

## Basic Git Commands You'll Use

### See What Changed
```bash
git status              # What files changed?
git diff                # Show exact changes
git log                 # See commit history
```

### Save Your Changes
```bash
git add filename.py     # Stage specific file
git add .               # Stage all changed files
git commit -m "Your commit message here"
```

### Undo Mistakes
```bash
git checkout filename.py    # Discard changes to a file
git reset HEAD filename.py  # Unstage a file
git revert HEAD             # Undo last commit (safe)
```

---

## Important: What Git WON'T Track

I've created a `.gitignore` file that excludes:

❌ **Model files** (`*.pkl`) - Too large (79 MB)
❌ **Dataset** (`*.csv`) - Too large (26 MB)  
❌ **Generated images** (`*.png`) - Can regenerate
❌ **Python cache** (`__pycache__/`) - Temporary files

✅ **Your code IS tracked**: `.py`, `.md`, `.txt` files

---

## Backing Up to GitHub (Optional)

### 1. Create GitHub Repository
- Go to https://github.com/new
- Name it "phishing-classifier"
- Don't initialize with README (you have one)

### 2. Link and Push
```bash
git remote add origin https://github.com/YOUR-USERNAME/phishing-classifier.git
git branch -M main
git push -u origin main
```

### 3. Future Updates
```bash
git add .
git commit -m "Updated feature extraction"
git push
```

---

## Real-World Workflow Example

### Scenario: You want to modify the prediction script

```bash
# 1. Make sure everything is committed first
git status

# 2. Make your changes to predict.py
# ... edit the file ...

# 3. See what changed
git diff predict.py

# 4. Save your changes
git add predict.py
git commit -m "Added batch size parameter to prediction"

# 5. If you pushed to GitHub
git push
```

### If You Made a Mistake
```bash
# Oops, that change broke something!
git log                    # Find the good commit hash
git checkout abc123        # Go back to that commit
# OR
git revert HEAD            # Undo last commit (cleaner)
```

---

## Common Questions

### Q: Do I HAVE to use Git?
**A:** No! Your classifier works fine without it. Git is just for tracking changes.

### Q: Why aren't model files tracked?
**A:** They're too large (79 MB). Git is designed for code, not binary files. You can track them separately or regenerate by running `train_model.py`.

### Q: What if I want to track the model anyway?
**A:** Use Git LFS (Large File Storage):
```bash
git lfs install
git lfs track "*.pkl"
git add .gitattributes
```

### Q: I accidentally committed something
**A:** 
```bash
git reset HEAD~1           # Undo last commit, keep changes
git reset --hard HEAD~1    # Undo last commit, discard changes
```

### Q: How do I see my project history?
**A:**
```bash
git log --oneline          # Compact view
git log --graph --oneline  # Visual tree
```

---

## Daily Git Workflow (Super Simple)

```bash
# End of day / after major changes:
git add .
git commit -m "Description of what you did today"

# If using GitHub:
git push
```

That's it! Three commands.

---

## Advanced: Branching (When You're Comfortable)

```bash
# Create a new branch for experiments
git branch experiment
git checkout experiment
# OR shorthand:
git checkout -b experiment

# Make changes, test, commit
git add .
git commit -m "Trying hyperparameter tuning"

# Switch back to main
git checkout main

# If experiment worked, merge it
git merge experiment

# If it didn't work, just delete the branch
git branch -d experiment
```

---

## Need Help?

```bash
git help                   # General help
git help commit            # Help for specific command
git status                 # When confused, start here
```

---

## Summary

**Minimum to get started:**
```bash
git init
git add .
git commit -m "Initial commit"
```

**Daily workflow:**
```bash
git add .
git commit -m "Your message"
git push    # If using GitHub
```

**When things go wrong:**
```bash
git status  # Always check this first
```

Git has a learning curve, but these basics will cover 90% of your needs!
