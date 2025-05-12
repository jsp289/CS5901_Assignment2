{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOscsfZrbrCz232QkadbwWp",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/jsp289/CS5901_Assignment2/blob/main/CS5901_assignment2_main_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CS5901 - Assignment 2 - Main Function**\n",
        "*This .py file coordinates function calls for the following with GitHub version control:*\n",
        "\n",
        "1.   Stage 1 - Loading CSV into Pandas DataFrame, Cleaning Data, and Basic Stats\n",
        "2.   Stage 2 - Calculating the Space-Time Complexity of Several Algorithms\n",
        "\n",
        "---\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "yyGrduo8mr7Z"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#git clone https://github.com/jsp289/CS5901_Assignment2.git #clone GitHub repo\n",
        "#%cd  CS5901_Assignment2 #change directory to repo\n",
        "\n",
        "from CS5901_assignment2_get_utils import get_git_log #other py file for GitHub repo ops\n",
        "from CS5901_assignment2_stage1_1_data_cleaning_py import import_data, remove_nonsensical_rows, replace_missing_values, calculate_stats, duplicate_rows # Stage 1 - Load CSV and Clean Data\n",
        "from CS5901_assignment2_stage2_time_space_complexity_py import complexity_analysis #Stage 2 - Complexity_analysis\n",
        "\n",
        "\n",
        "\n",
        "def run_demo(file_path, github_owner, github_repo, github_token = None):\n",
        "    \"\"\"\n",
        "    Executes Stages 1 & 2 of the Demo module with GitHub version control\n",
        "\n",
        "    Args:\n",
        "      file_path (str): url to the csv file\n",
        "      github_owner (str): owner of the GitHub repo\n",
        "      github_repo (str): name of the GitHub repo\n",
        "      github_token (str): GitHub personal access token\n",
        "\n",
        "    Returns:\n",
        "      dict: results of stages 1 & 2 with associated commit messages\n",
        "    \"\"\"\n",
        "\n",
        "    # Stage 0: Get GitHub commit history\n",
        "    git_log = get_git_log(github_owner, github_repo, github_token)\n",
        "\n",
        "    #Stage 1: Loading CSV and Cleaning Data\n",
        "    df = import_data(file_path)\n",
        "    df = remove_nonsensical_rows(df)\n",
        "    df = replace_missing_values(df)\n",
        "    calculate_stats(df)\n",
        "    duplicate_rows(df)\n",
        "\n",
        "    #Stage 2: Space & Time Complexity Analysis\n",
        "    complexity_results, sizes = complexity_analysis()\n",
        "\n",
        "    #Get version control history\n",
        "    #git_log = get_git_log(working_dir)\n",
        "\n",
        "    #Return results\n",
        "\n",
        "    return {\n",
        "        \"clean_data\": df_filled,\n",
        "        \"manual_stats\": manual_stats,\n",
        "        \"pandas_stats\": pd_stats,\n",
        "        \"duplicates\": (duplicate_data,df_filled_no_duplicates),\n",
        "        \"complexity\": (complexity_results,sizes),\n",
        "        \"git_log\": git_log,\n",
        "    }\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "\n",
        "  #GitHub personal info\n",
        "  owner = \"jsp289\" #replace with your own\n",
        "  repo = \"jsp289/CS5901_Assignment2\"\n",
        "  token = None # Replace with personal token if necessary\n",
        "\n",
        "  results = run_demo('https://raw.githubusercontent.com/jsp289/CS5901_Assignment2/refs/heads/main/P2data6332.csv', owner, repo, token)\n",
        "  #Verification output\n",
        "  print(\"Data loaded and cleaned successfully.\")\n",
        "  print(f\"Duplicates: {results['duplicates']}\")\n",
        "  print(f\"Manual Stats: {results['manual_stats']}\")\n",
        "  print(f\"Pandas Stats: {results['pandas_stats']}\")\n",
        "  print(f\"Complexity Results: {results['complexity_results']}\")"
      ],
      "metadata": {
        "id": "0kF3p_VZnjSi",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 384
        },
        "outputId": "2c04999b-35ba-422a-9b97-2a2a07549f82"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "No module named 'CS5901_assignment2_get_utils'",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-14-7c9e3e73aa17>\u001b[0m in \u001b[0;36m<cell line: 0>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0;31m#%cd  CS5901_Assignment2 #change directory to repo\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mCS5901_assignment2_get_utils\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mget_git_log\u001b[0m \u001b[0;31m#other py file for GitHub repo ops\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mCS5901_assignment2_stage1_1_data_cleaning_py\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mimport_data\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mremove_nonsensical_rows\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mreplace_missing_values\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcalculate_stats\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mduplicate_rows\u001b[0m \u001b[0;31m# Stage 1 - Load CSV and Clean Data\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0mCS5901_assignment2_stage2_time_space_complexity_py\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mcomplexity_analysis\u001b[0m \u001b[0;31m#Stage 2 - Complexity_analysis\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'CS5901_assignment2_get_utils'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "35224e3a"
      },
      "source": [
        "# from CS5901_assignment2_get_utils_py_py import get_git_log #other py file for GitHub repo ops\n",
        "from CS5901_assignment2_get_utils import get_git_log #other py file for GitHub repo ops\n",
        "from CS5901_assignment2_stage1_1_data_cleaning_py import import_data, remove_nonsensical_rows, replace_missing_values, calculate_stats, duplicate_rows # Stage 1 - Load CSV and Clean Data\n",
        "from CS5901_assignment2_stage2_time_space_complexity_py import complexity_analysis #Stage 2 - Complexity_analysis\n",
        "\n",
        "\n",
        "\n",
        "def run_demo(file_path, github_owner, github_repo, github_token = None):\n",
        "    \"\"\"\n",
        "    Executes Stages 1 & 2 of the Demo module with GitHub version control\n",
        "\n",
        "    Args:\n",
        "      file_path (str): url to the csv file\n",
        "      github_owner (str): owner of the GitHub repo\n",
        "      github_repo (str): name of the GitHub repo\n",
        "      github_token (str): GitHub personal access token\n",
        "\n",
        "    Returns:\n",
        "      dict: results of stages 1 & 2 with associated commit messages\n",
        "    \"\"\"\n",
        "\n",
        "    # Stage 0: Get GitHub commit history\n",
        "    git_log = get_git_log(github_owner, github_repo, github_token)\n",
        "\n",
        "    #Stage 1: Loading CSV and Cleaning Data\n",
        "    df = import_data(file_path)\n",
        "    df_no_nonsensical = remove_nonsensical_rows(df)\n",
        "    df_filled = replace_missing_values(df_no_nonsensical)\n",
        "    manual_stats, pd_stats = calculate_stats(df_filled)\n",
        "    duplicate_data, df_filled_no_duplicates = duplicate_rows(df_filled)\n",
        "\n",
        "\n",
        "    #Stage 2: Space & Time Complexity Analysis\n",
        "    complexity_results, sizes = complexity_analysis()\n",
        "\n",
        "    #Get version control history\n",
        "    # working_dir = 'CS5901_Assignment2' # Define the working directory\n",
        "    # git_log = get_git_log(working_dir)\n",
        "\n",
        "    #Return results\n",
        "\n",
        "    return {\n",
        "        \"clean_data\": df_filled,\n",
        "        \"manual_stats\": manual_stats,\n",
        "        \"pandas_stats\": pd_stats,\n",
        "        \"duplicates\": (duplicate_data,df_filled_no_duplicates),\n",
        "        \"complexity\": (complexity_results,sizes),\n",
        "        \"git_log\": git_log,\n",
        "    }\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "\n",
        "  #GitHub personal info\n",
        "  owner = \"jsp289\" #replace with your own\n",
        "  repo = \"jsp289/CS5901_Assignment2\"\n",
        "  token = None # Replace with personal token if necessary\n",
        "\n",
        "  # Use the direct URL to the CSV file on GitHub\n",
        "  results = run_demo('https://raw.githubusercontent.com/jsp289/CS5901_Assignment2/refs/heads/main/P2data6332.csv', owner, repo, token)\n",
        "  #Verification output\n",
        "  print(\"Data loaded and cleaned successfully.\")\n",
        "  # Accessing the results from the dictionary returned by run_demo\n",
        "  print(f\"Duplicates: {results['duplicates']}\")\n",
        "  print(f\"Manual Stats: {results['manual_stats']}\")\n",
        "  print(f\"Pandas Stats: {results['pandas_stats']}\")\n",
        "  print(f\"Complexity Results: {results['complexity']}\")"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}