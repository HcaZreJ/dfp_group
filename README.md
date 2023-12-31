# WeBall Fantasy Drafter

Welcome to WeBall Fantasy Drafter - the revolutionary tool that will transform your approach to fantasy basketball leagues! This cutting-edge platform is meticulously designed to assist both novice and seasoned users in assembling a dream team of NBA players. Let's dive into the incredible features that make WeBall a game-changer.

YOUTUBE: https://youtu.be/k2ge9lyKo0A?feature=shared

CMU Fall 2023, Data Focused Python Session D2, Group Project Team 3

## Features

### 1. Data Sources
WeBall taps into three distinct data sources to provide you with the most comprehensive player information:
- **NBA API:** Detailed player statistics for accurate and up-to-date information.
- **ESPN Website:** Latest news, player performance insights, and critical updates.
- **Spotrac Website:** Injury information to ensure your fantasy team is healthy and competitive.

### 2. Data Processing
The magic of WeBall lies in its advanced data processing capabilities. The model meticulously cleans, normalizes, and transforms gathered data, presenting you with a refined and comprehensive set of player details. Highly correlated parameters are removed, enhancing efficiency for a streamlined selection process.

### 3. Classifier Model
At the heart of WeBall Fantasy Drafter is its intelligent classifier model. This system employs sophisticated algorithms on the refined data to identify the absolute best players for your fantasy team. It's like having a personal basketball analyst tirelessly working to uncover hidden gems that will lead your team to victory.

## Installation Requirements

To get started with WeBall Fantasy Drafter, ensure you have Anaconda and pip installed. If you downloaded the zip file of the entire project, then navigate to the root folder of this project in your terminal, using something like:

```
cd /directory_to_WeBall_Fantasy_Drafter_root_dir
```

Alternatively, if you would like to clone the repository completely, then run the following commands (the first command is used to navigate to the directory where you would like the project to be in, while the second command is for cloning the repo):

```
cd /desired_directory_to_WeBall_Fantasy_Drafter
git clone https://github.com/HcaZreJ/dfp_group.git
```

now that you have the code on your pc, run this command:

```
conda env create -f environment.yml
```

This will create a Conda environment for you, under the name of `dfp_final`, and automatically install most of the necessary Python packages for you using Anaconda.

After the installations are all finished, you should run this command to switch to this environment:

```
conda activate dfp_final
```

After you activated the environment, you also need to install some packages that cannot be installed by Anaconda using pip. Just run:

```
pip install -r requirements.txt
```

Then you just need to run the `main.py` for the program to run!!

```
python main.py
```

Enjoy!

## Usage

1. Run only `main.py`.
2. The code will check for the presence of required data.
3. If data is not present, the code will automatically download it.
4. If data is present, the model will prompt the user to decide whether fresh data needs to be downloaded.
5. The model will also ask whether a young team or older team members are preferred.

## Contact Information

Feel free to reach out to the creators of WeBall Fantasy Drafter for any inquiries or support:
- Aditya Kolpe: akolpe@andrew.cmu.edu
- Zichen Zhu: zichenzh@andrew.cmu.edu
- Sophie Golunova: sgolunov@andrew.cmu.edu
- Brianna Dincau: bdincau@andrew.cmu.edu
- Emily Harvey: eharvey2@andrew.cmu.edu

Thank you for choosing WeBall Fantasy Drafter! Transform your fantasy basketball experience today.
