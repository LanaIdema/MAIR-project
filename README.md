# MAIR-project

## Setup
Install uv: https://docs.astral.sh/uv/getting-started/installation/

    uv sync
    # enable env
    . .venv/bin/activate

## CLI
To use the CLI, make sure to follow the following steps

    # Make sure to have the venv enabled 
    
    # Make sure the data is available
    python3 data_preprocessing.py
    
    # Make sure to generate the included models
    python3 evaluation.py

    # Open CLI tool
    python3 cli.py

## Git strategy (contributors)

*Branch strategy:*
- "main"-branch always contains the latest "official" code: this means there should be no bugs or errors in this code and every piece of code is final (at least for that version of the code)
- "[your-name]"-branch is your playground to write code and develop new features. It is wise to keep it as synced up as possible compared to "main".
- Youre free to make any other branches as feature branches, but that shouldnt really be necessary for this project

*Get your code up-to-date*
When you want to get started writing on a piece of code, always sync up your code with what the latest "official" version on the main branch. Other people may have pushed some changes in the time that you were away. To do this you should make sure youre on your branch and first commit your own changes on your own branch

    $ git checkout [your-branch]
    $ git add --all
    $ git commit -m [message]

Then you want to switch to the "main" branch and pull the changes

    $ git checkout main
    $ git pull

Switch back yo your own branch and get your branch up-to-date

    $ git checkout [your-branch]
    $ git merge main

If there are not many changes, you may instead want to use

    $ git checkout [your-branch]
    $ git rebase main

Make sure to resolve any conflicts you may have. Most of the time, you want to keep both changes. Sometimes you want to only keep the changes from main. Almost never, you want to overwrite main with your own changes, this can only happen when your absolutely certain that your code is improving the "official" code.

*Working your code*
If youre writing on a piece of code, make sure to not make one big commit but split your work up in commits that are one new feature. This is to make sure that if there are any merge conflicts, it is easy to see what commits are not compatible. This makes things easy to sort out when a whole merge goes south.

Youre also improving the chance that code actually merges by keeping your code-to-date by applying the strategy above.

*When your code is done*
Stage your changes

    $ git add --all

Make a commit

    $ git commit -m [what are you committing short description]

*To keep your remote nice and clean, make sure to sync changes from main now to your own branch as described above*
Push your code to the remote (so that it is not only locally available to you)

    $ git push --force

Force is to overwrite your branch on the remote (it is nice to know your own remote is the same as you have locally)

Then you can browse to github to create a pull request...