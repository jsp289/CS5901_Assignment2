{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPZP0h1Bo5r+zD33/xwRmG/",
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
        "<a href=\"https://colab.research.google.com/github/jsp289/CS5901_Assignment2/blob/main/CS5901_assignment2_stage2_time_space_complexity_py.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **CS5901 - Assignment 2 - Stage 2**\n",
        "*This .py file provides functions to calculate the time and space complexity for:*\n",
        "\n",
        "1.   Standard Matrix Multiplication\n",
        "2.   Unordered List of Integers\n",
        "3.   Substring Find() Method vs. Loop italicized text\n",
        "\n",
        "---\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "Gtq8yvvBADhr"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Stage 2.1** - Standard Matrix Multiplication Space-Time Complexity\n",
        "*Here we multiply a random integer matrix by a random float matrix of increasing size and compare their time and space complexities using a scatter plot.*"
      ],
      "metadata": {
        "id": "g-nyQ4zuBZA1"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Import all libraries required for space-time complexity analysis\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import time\n",
        "import gc\n",
        "import psutil\n",
        "import os\n",
        "import sys\n",
        "import string\n",
        "import random\n",
        "\n",
        "def matrix_multiplication(size):\n",
        "  \"\"\"\n",
        "  This function identifies multiplies 2 square matrices and computes space and time required to compute.\n",
        "\n",
        "  Args:\n",
        "    size (int): size of the square matrix\n",
        "  Returns:\n",
        "    run_time (float): time required to compute\n",
        "    mem_usage (float): space required to compute\n",
        "  \"\"\"\n",
        "  #Initiate random integer matrices\n",
        "  A = np.random.randint(1, 50000, size=(size, size))\n",
        "  B = np.random.randint(1, 50000, size=(size, size))\n",
        "\n",
        "  # Initiate space & time baselines\n",
        "  start_time = time.time()\n",
        "  process = psutil.Process(os.getpid())\n",
        "  base_mem = process.memory_info().rss\n",
        "\n",
        "  # Matrix multiplication\n",
        "  final_mat = np.dot(A, B)\n",
        "\n",
        "  #Compute running time and memory usage\n",
        "  end_time = time.time()\n",
        "  mem_usage = process.memory_info().rss - base_mem\n",
        "  run_time = end_time - start_time\n",
        "\n",
        "  return run_time, mem_usage\n",
        "print(matrix_multiplication(400))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ydSnRLOpKlP-",
        "outputId": "7e4a3eb2-bbe3-4c46-dd14-f06adbdf1144"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(0.07674121856689453, 1077248)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Stage 2.2** - Ordering Integers\n",
        "*Here we have a matrix of unordered integers, we flatten it, sort the elements from smallest to largest, and compute the space and time complexity of the sorting process*"
      ],
      "metadata": {
        "id": "L8IyFSbv4HAt"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def inneficient_integer_sort(size):\n",
        "  \"\"\"\n",
        "  This function generates a random matrix of integers, stores it in a list, and sorts them in ascending order.\n",
        "\n",
        "  Args:\n",
        "    size (int): size of the square matrix\n",
        "  Returns:\n",
        "    run_time (float): time required to compute\n",
        "    mem_usage (float): space required to compute\n",
        "  \"\"\"\n",
        "  #Generate random matrix and store in list\n",
        "  lst_rand_ints = np.random.randint(1,50000,size).tolist()\n",
        "  start = time.time()\n",
        "  process = psutil.Process(os.getpid())\n",
        "  base_mem = process.memory_info().rss\n",
        "\n",
        "  #Inneficient sort\n",
        "  sorted_lst_rand_ints = []\n",
        "  while lst_rand_ints:\n",
        "    min = lst_rand_ints[0]\n",
        "    for rand_int in lst_rand_ints:\n",
        "      if rand_int < min:\n",
        "        min = rand_int\n",
        "    sorted_lst_rand_ints.append(min)\n",
        "    lst_rand_ints.remove(min)\n",
        "\n",
        "  #Compute space and time complexity of inneficient sort\n",
        "  end = time.time()\n",
        "  mem_usage = process.memory_info().rss - base_mem\n",
        "  run_time = end - start\n",
        "\n",
        "  return run_time, mem_usage\n",
        "print(inneficient_integer_sort(5000))"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IKESAOGX5G_t",
        "outputId": "86cd171e-11a9-4b49-f1e9-fe98f182f4b0"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(0.19204092025756836, 0)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Stage 2.3** - String Find Method vs Manual Method\n",
        "*Here we generate a random string and compare the space-time complexity of manual sort vs the string.find() method*"
      ],
      "metadata": {
        "id": "QHd1yJ0u96ji"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def string_search_comparison(text,char):\n",
        "  \"\"\"\n",
        "  This function compares time of manual string search vs the string.find() method\n",
        "\n",
        "  Args:\n",
        "    text (str): a random string\n",
        "    char (str): a random character or substring\n",
        "  Returns:\n",
        "    str_find_time (float): time required to compute string.find() method\n",
        "    manual_find_time (float): time required to compute manual search\n",
        "\n",
        "  \"\"\"\n",
        "  # Str.find() time\n",
        "  start_find = time.time()\n",
        "  find_result = text.find(char)\n",
        "  end_find = time.time()\n",
        "  str_find_time = end_find - start_find\n",
        "\n",
        "  # Manual search time (the loop breaks when it finds the substring)\n",
        "  start_manual = time.time()\n",
        "  manual_count = -1\n",
        "  for i in range(len(text)-len(char)+1):\n",
        "    if text[i:i + len(char)] == char:\n",
        "      manual_count =1\n",
        "      break\n",
        "  end_manual = time.time()\n",
        "  manual_find_time = end_manual - start_manual\n",
        "\n",
        "  return str_find_time, manual_find_time\n",
        "\n",
        "\n",
        "text = \"a\" * 10000 + \"b\" + \"ab\" * 10000\n",
        "char = \"ab\"\n",
        "find_t, manual_t = string_search_comparison(text, char)\n",
        "print(f\"String.find() method time: {find_t}\")\n",
        "print(f\"Manual search time: {manual_t}\")\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZO4U-nhU_y__",
        "outputId": "7eecdc2e-77b1-4b88-f216-3e437e0f8e9b"
      },
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "String.find() method time: 7.867813110351562e-06\n",
            "Manual search time: 0.0009448528289794922\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "### **Stage 2.4** - Summary Analysis\n",
        "*Here we analyze the space-time complexity for all algorithms implemented for stages 2.1 to 2.3*"
      ],
      "metadata": {
        "id": "G_gREaTJuIpX"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def complexity_analysis(sizes=[100,500,1000]):\n",
        "  \"\"\"\n",
        "  This function compares time of manual string search vs the string.find() method\n",
        "\n",
        "  Args:\n",
        "    sizes (lst): a list of input sizes for testing\n",
        "\n",
        "  Returns:\n",
        "    tuple (results_dict, sizes_tested)\n",
        "\n",
        "  \"\"\"\n",
        "  #Initiate dictionary to store test results\n",
        "\n",
        "  results = {'matrix_mult':{'time':[],'space':[]},\n",
        "             'integer_sort':{'time':[],'space':[]}}\n",
        "\n",
        "  for size in sizes:\n",
        "    #clear memory before each test\n",
        "    gc.collect()\n",
        "\n",
        "    #test for matrix multiplixation\n",
        "    t,s = matrix_multiplication(size)\n",
        "    results['matrix_mult']['time'].append(t)\n",
        "    results['matrix_mult']['space'].append(s)\n",
        "\n",
        "    #test for inneficient integer sort\n",
        "    t,s = inneficient_integer_sort(size)\n",
        "    results['integer_sort']['time'].append(t)\n",
        "    results['integer_sort']['space'].append(s)\n",
        "\n",
        "  #test for string search\n",
        "  text = \"a\" * 10000 + \"b\" + \"ab\" * 10000\n",
        "  char = \"ab\"\n",
        "  str_find_time, manual_find_time = string_search_comparison(text, char)\n",
        "  results['string_search'] = {'time':[str_find_time, manual_find_time]}\n",
        "\n",
        "  return results, sizes\n",
        "\n",
        "# Call and print results\n",
        "results, sizes = complexity_analysis(sizes=[100, 500, 1000])\n",
        "print(\"\\nComplexity Analysis Results:\")\n",
        "for algorithm, data in results.items():\n",
        "    print(f\"\\n{algorithm}:\")\n",
        "    for metric, values in data.items():\n",
        "        print(f\"  {metric}: {values}\")\n",
        "print(\"\\nSizes Tested:\", sizes)\n",
        "\n",
        "# Create scatter plots\n",
        "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns\n",
        "\n",
        "# Time Complexity Plot\n",
        "ax1.scatter(sizes, results['matrix_mult']['time'], label='Matrix Multiplication')\n",
        "ax1.scatter(sizes, results['integer_sort']['time'], label='Integer Sort')\n",
        "ax1.set_xlabel('Input Size')\n",
        "ax1.set_ylabel('Time (seconds)')\n",
        "ax1.set_title('Time Complexity')\n",
        "ax1.legend()\n",
        "\n",
        "# Space Complexity Plot\n",
        "ax2.scatter(sizes, results['matrix_mult']['space'], label='Matrix Multiplication')\n",
        "ax2.scatter(sizes, results['integer_sort']['space'], label='Integer Sort')\n",
        "ax2.set_xlabel('Input Size')\n",
        "ax2.set_ylabel('Space (bytes)')\n",
        "ax2.set_title('Space Complexity')\n",
        "ax2.legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n"
      ],
      "metadata": {
        "id": "kKbDsB5Kvq_u",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 867
        },
        "outputId": "bbddf920-1413-4238-95cc-5c270457e52d"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Complexity Analysis Results:\n",
            "\n",
            "matrix_mult:\n",
            "  time: [0.0012919902801513672, 0.16057157516479492, 1.0748181343078613]\n",
            "  space: [0, 1892352, 8228864]\n",
            "\n",
            "integer_sort:\n",
            "  time: [0.00023245811462402344, 0.0035381317138671875, 0.01324009895324707]\n",
            "  space: [0, 0, 0]\n",
            "\n",
            "string_search:\n",
            "  time: [1.7642974853515625e-05, 0.0014843940734863281]\n",
            "\n",
            "Sizes Tested: [100, 500, 1000]\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1200x600 with 2 Axes>"
            ],
            "image/png": "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CxcuTLPtzJkzunLliv15165dlZqaqs8//1yffvqpPDw81KVLl1t+81igQIE0jfA1Hh4eatOmjb7++mtNnTpVNWrUUM2aNW+6PwB5T48ePRQbG6tZs2bp999/19NPP62mTZva/4H+448/qkKFCpo/f77Kly+v4OBgPf/885wphWxv6dKl6f4dvbYe5D+XAYiNjXVYF+rw4cOaN2+eHn74YfsZQu7u7mn2+eGHH9rXnLrmySef1NatW/Xdd9+lef9rr3/mmWf0999/a9KkSWnmXLx4UefPn7/p8UVHR+vkyZN6/vnnHXqKaxYtWqT58+dLkpo3b664uDjNnj3bvv3KlSv68MMPVbBgQaevmXTkyBGHY09MTNT06dNVu3btG57NVKdOHVWsWFHvvPOOzp07l2b79b3kgQMH9PLLL+vJJ5/Uq6++qnfeeUc//PCDpk+fftO6ChQoIEk37J2aNWum4sWLa8yYMVq+fDlnSQGZwJlSAFzK19dXEyZMUPv27XX33Xfr2WefVYkSJXTo0CH99NNPatCggcaPH3/D1xcpUkTfffedmjdvrtq1a6tdu3b2xUY3bdqkr776yn45W4kSJTRo0CBFR0eradOmeuSRR7R79259/PHHqlevXpY0EAcOHNAjjzyipk2bKjY2Vl9++aWee+451apV64avefnll/XDDz+oZcuW6tixo+rUqaPz589r27Ztmjt3rv766y8VL15cU6ZM0U8//aSpU6eqTJkykq42uO3atdOECRPUrVu3G75HnTp1NGHCBL355puqVKmS/P39HdaMioqK0gcffKClS5dqzJgxzvuBAMgVDh06pClTpujQoUMqVaqUJKl///5asGCBpkyZopEjR2r//v06ePCg5syZo+nTpyslJUV9+/bVU089pSVLlrj4CIAb69mzpy5cuKDHH39cVatWVXJyslavXq3Zs2crODhYnTp1cpgfGhqqyMhI9erVS15eXvr4448lXQ1/rmnZsqW++OIL+fn5KSQkRLGxsfr1119VrFgxh329/PLLmjt3rp5++ml17txZderU0alTp/TDDz9o4sSJqlWrltq3b6+vv/5aL774opYuXaoGDRooJSVFu3bt0tdff62FCxfedOHu1q1ba9u2bRoxYoQ2b96sNm3aqFy5cjp58qQWLFigmJgY+9qSL7zwgj755BN17NhRGzduVHBwsObOnatVq1Zp3LhxGV4wPaOqVKmiLl26aP369QoICNDkyZN17NgxTZky5YavcXNz02effaZmzZqpevXq6tSpk0qXLq2///5bS5cula+vr3788UcZY9S5c2f5+PhowoQJkqR///vf+uabb9S7d281btzY/vvsnypWrKjChQtr4sSJKlSokAoUKKDw8HD72Z/58uXTs88+q/Hjx8vd3f2WZ+8DuI6L7voHIBvr3r27udGvhw4dOphy5crZn1+7BfDbb7/tMO/a7XrnzJnjMD5lyhQjKc2td5cuXWoiIyONn5+f8fb2NhUrVjQdO3Z0uMXyzRw5csT07dvXVKlSxXh7e5v8+fObOnXqmBEjRpiEhASHuePHjzdVq1Y1+fLlMwEBAeall14yp0+fdphz//33m+rVq6d5n3LlypkWLVqkGZdkunfvbn8+dOhQI8ns2LHDPPXUU6ZQoUKmSJEipkePHubixYtp9tmhQweHsbNnz5pBgwaZSpUqGU9PT1O8eHFTv359884775jk5GRz+PBh4+fnZ1q1apWmlscff9wUKFDA7N+/3xjzv5/5gQMH7HPi4uJMixYtTKFChYwkh9seX1O9enXj5uZm/vvf/6bZBiBvkWS+++47+/P58+cbSaZAgQIODw8PD/PMM88YY4zp2rWrkWR2795tf93GjRuNJLNr1y6rDwHIsF9++cV07tzZVK1a1RQsWNB4enqaSpUqmZ49e5pjx445zL329//LL780lStXNl5eXiYsLMwsXbrUYd7p06dNp06dTPHixU3BggVNZGSk2bVrV7o9wMmTJ02PHj1M6dKljaenpylTpozp0KGDOXHihH1OcnKyGTNmjKlevbrx8vIyRYoUMXXq1DHR0dFp+p4biYmJMY8++qjx9/c3Hh4epkSJEqZVq1Zm3rx5DvOOHTtmr93T09PUqFHDTJkyxWGOM/rBaz3WwoULTc2aNY2Xl5epWrVqmtde2+c/f8abN282TzzxhClWrJjx8vIy5cqVM88884yJiYkxxhjz/vvvG0nmm2++cXjdoUOHjK+vr2nevLl97P7770/TG82bN8+EhIQYDw8PIynNz2DdunVGknn44YcNgIyzGeOE1eUAAA6GDRum6OhoHT9+XMWLF3d1ObclLCxMRYsWVUxMjKtLAeBiNptN3333nR577DFJ0uzZs9W2bVv98ccfaRYwLliwoAIDAzV06NA0l2ZfvHhR+fPn16JFi9SkSRMrDwHIEjabTd27d7/pWd3ImODgYIWGhtovHcxptm7dqtq1a2v69OmsxQlkApfvAQDS2LBhg7Zs2aKpU6e6uhQA2VBYWJhSUlIUHx+v++67L905DRo00JUrV7Rv3z77QsTX1v5z9uLIAOBqkyZNUsGCBW/7bqVAXkUoBQCw2759uzZu3Kh3331XJUuWVOvWrV1dEgAXOXfunPbu3Wt/fuDAAW3ZskVFixZVlSpV1LZtW0VFRendd99VWFiYjh8/rpiYGNWsWVMtWrRQ48aNdffdd6tz584aN26cUlNT1b17dzVp0kRVqlRx4ZEBgPP8+OOP2rFjhz799FP16NHDvig6gIzh7nsAALu5c+eqU6dOunz5sr766it5e3u7uiQALrJhwwaFhYUpLCxMktSvXz+FhYVpyJAhkqQpU6YoKipK//nPf3TXXXfpscce0/r161W2bFlJVxcf/vHHH1W8eHE1atRILVq0ULVq1TRr1iyXHRMAOFvPnj01bNgwNW/e3GFxewAZw5pSAAAAAAAAsBxnSgEAAAAAAMByhFIAAAAAAACwXJ5b6Dw1NVVHjhxRoUKFZLPZXF0OAADIoYwxOnv2rEqVKiU3t9zxPR99EgAAcIaM9kl5LpQ6cuSIgoKCXF0GAADIJQ4fPqwyZcq4ugynoE8CAADOdKs+Kc+FUoUKFZJ09Qfj6+vr4moAAEBOlZiYqKCgIHtvkRvQJwEAAGfIaJ+U50Kpa6ei+/r60mwBAIA7lpsuc6NPAgAAznSrPil3LIAAAAAAAACAHIVQCgAAAAAAAJYjlAIAAAAAAIDl8tyaUhmVkpKiy5cvu7oMwBL58uWTu7u7q8sAAOQQ9EnIS+iTACDrEEr9gzFGcXFxOnPmjKtLASxVuHBhBQYG5qoFewEAzkWfhLyKPgkAsgah1D9ca7T8/f2VP39+/vAg1zPG6MKFC4qPj5cklSxZ0sUVAQCyK/ok5DX0SQCQtQilrpOSkmJvtIoVK+bqcgDL+Pj4SJLi4+Pl7+/PKeoAgDTok5BX0ScBQNZhofPrXFsbIX/+/C6uBLDetf/uWSMEAJAe+iTkZfRJAJA1CKXSwanoyIv47x4AkBH8vUBexH/3AJA1CKUAAAAAAABgOUIpOEVwcLDGjRvn6jIcdOzYUY899thN5yxbtkw2my1TdxEaNmyYateunan3cQabzabvv/8+y98HAAA4F33Szd/HGeiTACBnIpTKJTp27CibzaYXX3wxzbbu3bvLZrOpY8eOGd7fX3/9JZvNpi1btmRo/vr16/XCCy9keP//dK3pKVKkiC5dupRm3zab7Y5Pm37ggQfUp08fh7H69evr6NGj8vPzu+39vv/++5o6deod1Xa9fzZz1xw9elTNmjVz2vsAAJBX0CfdGn0SAMAVCKWySEqqUey+k5q35W/F7juplFST5e8ZFBSkWbNm6eLFi/axS5cuaebMmSpbtmyWvGdycrIkqUSJEk5Z+LRQoUL67rvvHMY+//zzLKvf09NTgYGBd9TI+fn5qXDhws4r6gYCAwPl5eWV5e8DAEBWo0+6PfRJN0afBAA5E6FUFliw/agajlmiNpPWqPesLWozaY0ajlmiBduPZun73n333QoKCtK3335rH/v2229VtmxZhYWFOda4YIEaNmyowoULq1ixYmrZsqX27dtn316+fHlJUlhYmGw2mx544AFJ/zsFe8SIESpVqpTuuusuSY6npS9btkyenp5asWKFfX9vvfWW/P39dezYsZseQ4cOHTR58mT784sXL2rWrFnq0KGDw7z0viUbN26cgoOD091vx44dtXz5cr3//vv2bxP/+uuvNKelT506VYULF9b333+vypUry9vbW5GRkTp8+PANa/7naempqal66623VKlSJXl5eals2bIaMWKEffuAAQNUpUoV5c+fXxUqVNDgwYPtd3KZOnWqoqOjtXXrVnud175d/Odp6du2bdO//vUv+fj4qFixYnrhhRd07ty5NHW98847KlmypIoVK6bu3btz1xgAgEvRJ9En0ScBAK4hlHKyBduP6qUvN+loguOp1XEJl/TSl5uyvOHq3LmzpkyZYn8+efJkderUKc288+fPq1+/ftqwYYNiYmLk5uamxx9/XKmpqZKkdevWSZJ+/fVXHT161KGBi4mJ0e7du7V48WLNnz8/zb6vnf7dvn17JSQkaPPmzRo8eLA+++wzBQQE3LT+9u3ba8WKFTp06JAk6ZtvvlFwcLDuvvvuzP8wrvP+++8rIiJCXbt21dGjR3X06FEFBQWlO/fChQsaMWKEpk+frlWrVunMmTN69tlnM/xegwYN0ujRozV48GDt2LFDM2fOdDjuQoUKaerUqdqxY4fe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XF2K5Zz13z8AuErsvpNqM2nNLed91fVepyykfbOeIqeiT8LN0CfRJwHIubJrn8Td97JA09CSahISmG1usQgAQF5w7fKwuIRL6a6XYJMUaMHlYbg5+iQAAKyXXfskQqks4u5my7LbGQMAgLSuXR720pebZJMcGi6rLw/DzdEnAQBgrezaJ7GmFHCdjh075slT0gEgt2gaWlIT2t2tQD/Hy2sC/bw1od3dXB4G3AH6JADI2bJjn8SZUgAAIFfh8jAAAID0Zbc+iVAKAADkOlweBgAAkL7s1Cdx+R4AAAAAAAAsRygFAAAAAAAAyxFKAQAA5BIpKSkaPHiwypcvLx8fH1WsWFHDhw+XMend/BkAAMC1WFMKAAAglxgzZowmTJigadOmqXr16tqwYYM6deokPz8/9erVy9XlAQAAOCCUAgAAyCVWr16tRx99VC1atJAkBQcH66uvvtK6detcXBkAAEBaXL6XS3Ts2FGPPfZYpl5js9n0/fffZ0k9zpKSkqLRo0eratWq8vHxUdGiRRUeHq7PPvvsjvc9bNgw1a5d+86LBAAgm6hfv75iYmL0559/SpK2bt2qlStXqlmzZunOT0pKUmJiosMjN6JPyjz6JACAFThTKqukpkgHV0vnjkkFA6Ry9SU3d1dXlW0lJyfL09MzzXh0dLQ++eQTjR8/XnXr1lViYqI2bNig06dP3/Z7GWOUkpJyJ+UCAJAtDRw4UImJiapatarc3d2VkpKiESNGqG3btunOHzVqlKKjoy2uUvRJmUSfBADIrThTKivs+EEaFypNayl90+Xq/x0XenXcIg888IB69eqlV155RUWLFlVgYKCGDRtm3x4cHCxJevzxx2Wz2ezPJWnevHm6++675e3trQoVKig6OlpXrlyxb9+1a5caNmwob29vhYSE6Ndff03zbeLhw4f1zDPPqHDhwipatKgeffRR/fXXX/bt176xHDFihEqVKqW77ror3eP44Ycf1K1bNz399NMqX768atWqpS5duqh///72OUlJSerVq5f8/f3l7e2thg0bav369fbty5Ytk81m0y+//KI6derIy8tLX375paKjo7V161bZbDbZbDZNnTr1tn7WAABkF19//bVmzJihmTNnatOmTZo2bZreeecdTZs2Ld35gwYNUkJCgv1x+PDhrC+SPok+CQCA/0co5Ww7fpC+jpISjziO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31HRAAAAec2QIUM0c+ZMXbx40dWlAAAAWCrTodTAgQPVt29frV27VmPHjtXYsWO1du1a9enTh8voAAAAMiksLEz9+/dXYGCgunbtqjVr1ri6JAAAAEtkOpTauXOnunTpkma8c+fO2rFjh1OKAgAAyCvGjRunI0eOaMqUKYqPj1ejRo0UEhKid955R8eOHXN1eQAAAFkm06FUiRIltGXLljTjW7Zskb+/vzNqAgAAyFM8PDz0xBNPaN68efrvf/+r5557ToMHD1ZQUJAee+wxLVmyxNUlAgAAOF2mV9Ts2rWrXnjhBe3fv1/169eXJK1atUpjxoxRv379nF4gAABAXrFu3TpNmTJFs2bNkr+/vzp27Ki///5bLVu2VLdu3fTOO++4ukQAAACnyfTd94wxGjdunN59910dOXJEklSqVCm9/PLL6tWrl2w2W5YU6izcVQYAADiDs3qK+Ph4ffHFF5oyZYr27NmjVq1a6fnnn1dkZKS9r1q5cqWaNm2qc+fOOav8dNEnAQAAZ8hoT5HpM6VsNpv69u2rvn376uzZs5KkQoUK3X6lAAAAeViZMmVUsWJFde7cWR07dlSJEiXSzKlZs6bq1avnguoAAACyTqZDqQMHDujKlSuqXLmyQxi1Z88e5cuXT8HBwc6sDwAAIFeLiYnRfffdd9M5vr6+Wrp0qUUVAQAAWCPTC5137NhRq1evTjO+du1adezY0Rk1AQAA5BnXAqn4+HitWLFCK1asUHx8vIurAgAAyHqZDqU2b96sBg0apBm/9957070rHwAAAG7s7Nmzat++vUqXLq37779f999/v0qXLq127dopISHB1eUBAABkmUyHUjabzb6W1PUSEhKUkpLilKIAAADyiueff15r167V/PnzdebMGZ05c0bz58/Xhg0b9O9//9vV5QEAAGSZTN99r1WrVvLx8dFXX30ld3d3SVJKSopat26t8+fP65dffsmSQp2Fu8oAAABncFZPUaBAAS1cuFANGzZ0GF+xYoWaNm2q8+fP32mpGUafBAAAnCHL7r43ZswYNWrUSHfddZd9DYQVK1YoMTFRS5Ysuf2KAQAA8qBixYrJz88vzbifn5+KFCnigooAAACskenL90JCQvT777/rmWeeUXx8vM6ePauoqCjt2rVLoaGhWVEjAABArvX666+rX79+iouLs4/FxcXp5Zdf1uDBg11YGQAAQNbK9JlSklSqVCmNHDnS2bUAAADkCWFhYbLZbPbne/bsUdmyZVW2bFlJ0qFDh+Tl5aXjx4+zrhQAAMi1biuUWrFihT755BPt379fc+bMUenSpfXFF1+ofPnyadZDAAAAgKPHHnvM1SUAAAC4XKZDqW+++Ubt27dX27ZttWnTJiUlJUm6eve9kSNH6ueff3Z6kQAAALnJ0KFDXV0CAACAy2V6Tak333xTEydO1KRJk5QvXz77eIMGDbRp0yanFgcAAJAbZfLmxwAAALlSpkOp3bt3q1GjRmnG/fz8dObMGWfUBAAAkKtVr15ds2bNUnJy8k3n7dmzRy+99JJGjx5tUWUAAADWyfTle4GBgdq7d6+Cg4MdxleuXKkKFSo4qy4AAIBc68MPP9SAAQPUrVs3NWnSRHXr1lWpUqXk7e2t06dPa8eOHVq5cqX++OMP9ejRQy+99JKrSwYAAHC6TIdSXbt2Ve/evTV58mTZbDYdOXJEsbGx6t+/P7ctBgAAyICHHnpIGzZs0MqVKzV79mzNmDFDBw8e1MWLF1W8eHGFhYUpKipKbdu2VZEiRVxdLgAAQJbIdCg1cOBApaam6qGHHtKFCxfUqFEjeXl5qX///urZs2dW1AgAAJArNWzYkDsXAwCAPCvTa0rZbDa99tprOnXqlLZv3641a9bo+PHjGj58eFbUBwAAgNswevRo2Ww29enTx9WlAAAApCvTodQ1np6eCgkJUdWqVfXrr79q586dzqwLAAAAt2n9+vX65JNPVLNmTVeXAgAAcEOZDqWeeeYZjR8/XpJ08eJF1atXT88884xq1qypb775xukFAgAAIOPOnTuntm3batKkSaxHBQAAsrVMh1K//fab7rvvPknSd999p9TUVJ05c0YffPCB3nzzTacXCAAAgIzr3r27WrRoocaNG7u6FAAAgJvK9ELnCQkJKlq0qCRpwYIFevLJJ5U/f361aNFCL7/8stMLBAAAQMbMmjVLmzZt0vr16zM0PykpSUlJSfbniYmJWVUaAABAGpk+UyooKEixsbE6f/68FixYoIcffliSdPr0aXl7ezu9QAAAgNxu3759ev3119WmTRvFx8dLkn755Rf98ccfGd7H4cOH1bt3b82YMSPDPdmoUaPk5+dnfwQFBd1W/QAAALcj06FUnz591LZtW5UpU0alSpXSAw88IOnqZX01atRwdn0AAAC52vLly1WjRg2tXbtW3377rc6dOydJ2rp1q4YOHZrh/WzcuFHx8fG6++675eHhIQ8PDy1fvlwffPCBPDw8lJKSkuY1gwYNUkJCgv1x+PBhpx0XAADArWT68r1u3bopPDxchw4dUpMmTeTmdjXXqlChAmtKAQAAZNLAgQP15ptvql+/fipUqJB9/F//+pf95jIZ8dBDD2nbtm0OY506dVLVqlU1YMAAubu7p3mNl5eXvLy8br94AACAO5DpUEqS6tSpozp16jiMtWjRwikFAQAA5CXbtm3TzJkz04z7+/vrxIkTGd5PoUKFFBoa6jBWoEABFStWLM04AABAdpChy/dGjx6tixcvZmiHa9eu1U8//XRHRQEAAOQVhQsX1tGjR9OMb968WaVLl3ZBRQAAANbI0JlSO3bsUNmyZfX000+rVatWqlu3rkqUKCFJunLlinbs2KGVK1fqyy+/1JEjRzR9+vQsLRoAACC3ePbZZzVgwADNmTNHNptNqampWrVqlfr376+oqKg72veyZcucUyQAAEAWyNCZUtOnT9evv/6qy5cv67nnnlNgYKA8PT1VqFAheXl5KSwsTJMnT1ZUVJR27dqlRo0aZXXdAAAAucLIkSNVtWpVBQUF6dy5cwoJCVGjRo1Uv359vf76664uDwAAIMvYjDEmMy9ITU3V77//roMHD+rixYsqXry4ateureLFi2dVjU6VmJgoPz8/JSQkyNfX19XlAACAHMrZPcXhw4e1bds2nTt3TmFhYapcubITqswc+iQAAOAMGe0pMr3QuZubm2rXrq3atWvfSX0AAAC4TlBQkIKCglxdBgAAgGUydPkeAAAAssaTTz6pMWPGpBl/66239PTTT7ugIgAAAGsQSgEAALjQb7/9pubNm6cZb9asmX777TcXVAQAAGANQikAAAAXOnfunDw9PdOM58uXT4mJiS6oCAAAwBqEUgAAAC5Uo0YNzZ49O834rFmzFBIS4oKKAAAArJHphc6v2bt3r/bt26dGjRrJx8dHxhjZbDZn1gYAAJDrDR48WE888YT27dunf/3rX5KkmJgYffXVV5ozZ46LqwMAAMg6mQ6lTp48qdatW2vJkiWy2Wzas2ePKlSooC5duqhIkSJ69913s6JOAACAXKlVq1b6/vvvNXLkSM2dO1c+Pj6qWbOmfv31V91///2uLg8AACDLZPryvb59+8rDw0OHDh1S/vz57eOtW7fWggULnFocAABAXtCiRQutWrVK58+f14kTJ7RkyRICKQAAkOtl+kypRYsWaeHChSpTpozDeOXKlXXw4EGnFQYAAAAAAIDcK9Oh1Pnz5x3OkLrm1KlT8vLyckpRAAAAeUVKSoree+89ff311zp06JCSk5Mdtp86dcpFlQEAAGStTF++d99992n69On25zabTampqXrrrbf04IMPOrU4AACA3C46Olpjx45V69atlZCQoH79+umJJ56Qm5ubhg0b5uryAAAAskymz5R666239NBDD2nDhg1KTk7WK6+8oj/++EOnTp3SqlWrsqJGAACAXGvGjBmaNGmSWrRooWHDhqlNmzaqWLGiatasqTVr1qhXr16uLhEAACBLZPpMqdDQUP35559q2LChHn30UZ0/f15PPPGENm/erIoVK2ZFjQAAALlWXFycatSoIUkqWLCgEhISJEktW7bUTz/95MrSAAAAslSmz5SSJD8/P7322mvOrgUAACDPKVOmjI4ePaqyZcuqYsWKWrRoke6++26tX7+e9ToBAECudluh1KVLl/T7778rPj5eqampDtseeeQRpxQGAACQFzz++OOKiYlReHi4evbsqXbt2unzzz/XoUOH1LdvX1eXBwAAkGUyHUotWLBAUVFROnHiRJptNptNKSkpTikMAAAgLxg9erT9f7du3Vply5ZVbGysKleurFatWrmwMgAAgKyV6VCqZ8+eevrppzVkyBAFBARkRU0AAAB5VkREhCIiIlxdBgAAQJbLdCh17Ngx9evXj0AKAADASXbv3q0PP/xQO3fulCRVq1ZNPXv21F133eXiygAAALJOpu++99RTT2nZsmVZUAoAAEDe88033yg0NFQbN25UrVq1VKtWLW3atEmhoaH65ptvXF0eAABAlrEZY0xmXnDhwgU9/fTTKlGihGrUqKF8+fI5bO/Vq5dTC3S2xMRE+fn5KSEhQb6+vq4uBwAA5FDO6ikqVqyotm3b6o033nAYHzp0qL788kvt27fvTkvNMPokAADgDBntKTJ9+d5XX32lRYsWydvbW8uWLZPNZrNvs9ls2T6UAgAAyE6OHj2qqKioNOPt2rXT22+/7YKKAAAArJHpUOq1115TdHS0Bg4cKDe3TF/9BwAAgOs88MADWrFihSpVquQwvnLl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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "dINMOGRifsM-"
      },
      "execution_count": 8,
      "outputs": []
    }
  ]
}