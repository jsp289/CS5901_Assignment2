{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "collapsed_sections": [
        "g-nyQ4zuBZA1",
        "L8IyFSbv4HAt",
        "QHd1yJ0u96ji",
        "G_gREaTJuIpX"
      ],
      "authorship_tag": "ABX9TyPZP0h1Bo5r+zD33/xwRmG/"
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
        "outputId": "fff37de5-9d8f-436d-e2bf-e2d9da584c8d"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(0.41768574714660645, 1081344)\n"
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
        "outputId": "eb496441-b2f1-46f1-b698-3ac1dfc6ddb4"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "(1.8223328590393066, 0)\n"
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
        "outputId": "1b11d32a-ad40-45ad-a1c2-1c9d37074e68"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "String.find() method time: 2.002716064453125e-05\n",
            "Manual search time: 0.011854410171508789\n"
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
        "outputId": "7bcb99f8-432d-4ed7-a4c0-92807c17879e"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Complexity Analysis Results:\n",
            "\n",
            "matrix_mult:\n",
            "  time: [0.0056400299072265625, 0.7152583599090576, 2.306227207183838]\n",
            "  space: [0, 1892352, 7901184]\n",
            "\n",
            "integer_sort:\n",
            "  time: [0.0004143714904785156, 0.0078012943267822266, 0.025463104248046875]\n",
            "  space: [0, 0, 0]\n",
            "\n",
            "string_search:\n",
            "  time: [2.2649765014648438e-05, 0.0027074813842773438]\n",
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
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABKUAAAJOCAYAAABm7rQwAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAAfFdJREFUeJzs3XlcFWX7x/HvAWRxATcQF9xwRVxwyVBTKxVNLVvNNHDJnkrLJTOtXKhMbbWy1KwkS7Os1LJyX1PclzTN1Eh9CtwFcUGF+/eHP87jCVTQwxyWz/v1mlfOPffMueYMHS6uc889NmOMEQAAAAAAAGAhN1cHAAAAAAAAgIKHohQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlAIAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAAAAAAABYjqIUgGzp2bOnKleu7Oow8pXKlSurZ8+eOXb8mJgY2Ww2/fXXXzn2GgAAAFZYsWKFbDabVqxYkWOv0bp1a7Vu3TrHjg/gfyhKAZDNZsvSkpO//G/W4cOHNWTIENWqVUuFCxdWkSJF1KhRI7366qs6deqUq8PLdT788EPFxMS4OgwAAHKlHTt26IEHHlClSpXk7e2t8uXLq23btnr//fddHZrTrFixQvfdd58CAwPl6empgIAAde7cWd99952rQ8t1/vnnH40ePVrbtm1zdShAvmMzxhhXBwHAtb744guH9enTp2vx4sX6/PPPHdrbtm2rkiVLKi0tTV5eXlaGeE0bN27UXXfdpeTkZPXo0UONGjWSJG3atEmzZs1Ss2bNtGjRIhdHeXWVK1dW69atc6xIlJqaqosXL8rLy0s2m02SFBoaqtKlS+fqQiMAAK6wdu1a3X777apYsaKioqIUGBioQ4cOad26ddq/f7/27dvn6hBv2qhRo/Tyyy+revXq6tatmypVqqTjx4/rp59+0ooVKzRjxgw98sgjrg4zUytWrNDtt9+u5cuX59hopgsXLkiSPD09JV3OKZs0aaJp06bl6Oh2oCDycHUAAFyvR48eDuvr1q3T4sWLM7TnRqdOndK9994rd3d3bd26VbVq1XLYPmbMGE2dOtVF0eUO7u7ucnd3d3UYAADkCWPGjJGfn582btyo4sWLO2w7cuSIa4Jyom+++UYvv/yyHnjgAc2cOVOFChWyb3vuuee0cOFCXbx40YURul56MQpAzuP2PQDZ8u85pf766y/ZbDa9+eab+uCDD1S1alUVLlxY7dq106FDh2SM0SuvvKIKFSrIx8dH99xzj06cOJHhuD///LNuu+02FSlSRMWKFVPHjh3122+/XTeeKVOm6O+//9bbb7+doSAlSWXKlNFLL73k0Pbhhx+qTp068vLyUrly5dSvX78Mt/i1bt1aoaGh+vXXX9WqVSsVLlxY1apV0zfffCNJWrlypZo2bSofHx/VrFlTS5Yscdh/9OjRstls+v333/XQQw/J19dXpUqV0oABA3T+/PnrntepU6c0cOBABQUFycvLS9WqVdP48eOVlpYmSTLG6Pbbb5e/v79DgnzhwgXVrVtXwcHBOnPmjKSMc0pVrlxZv/32m1auXGm/NbN169b6888/ZbPZ9M4772SIZ+3atbLZbPryyy+vGzsAAHnZ/v37VadOnQwFKUkKCAhwWLfZbOrfv79mzJihmjVrytvbW40aNdKqVasc+h04cEBPPfWUatasKR8fH5UqVUoPPvhgpvM9njp1SoMGDVLlypXl5eWlChUqKDIyUseOHbP3SUlJ0ahRo1StWjV5eXkpKChIQ4cOVUpKynXPb8SIESpZsqQ+/fRTh4JUuoiICHXq1Mm+fuTIEfXp00dlypSRt7e36tevr88++8xhH2fkg5UrV1anTp20aNEiNWjQQN7e3goJCcny7YTr169X+/bt5efnp8KFC6tVq1Zas2aNffvu3bvl4+OjyMhIh/1++eUXubu76/nnn7e3XTmn1IoVK9SkSRNJUq9evey5U0xMjEaNGqVChQrp6NGjGeJ5/PHHVbx48SzlfUCBZgDgX/r162eu9vEQFRVlKlWqZF+Pi4szkkyDBg1MSEiIefvtt81LL71kPD09za233mpeeOEF06xZM/Pee++ZZ555xthsNtOrVy+HY06fPt3YbDbTvn178/7775vx48ebypUrm+LFi5u4uLhrxtqsWTPj4+NjUlJSsnRuo0aNMpJMmzZtzPvvv2/69+9v3N3dTZMmTcyFCxfs/Vq1amXKlStngoKCzHPPPWfef/99ExISYtzd3c2sWbNMYGCgGT16tJkwYYIpX7688fPzM0lJSRlep27duqZz585m4sSJpkePHkaSefTRRx1iqlSpkomKirKvnzlzxtSrV8+UKlXKvPDCC2by5MkmMjLS2Gw2M2DAAHu/P//80xQtWtTce++99rZhw4YZm81mVq5caW+bNm2akWR/L+fMmWMqVKhgatWqZT7//HPz+eefm0WLFhljjGnevLlp1KhRhvftqaeeMsWKFTNnzpzJ0vsMAEBe1a5dO1OsWDGzY8eO6/aVZEJDQ03p0qXNyy+/bMaPH28qVapkfHx8HPafPXu2qV+/vhk5cqT56KOPzAsvvGBKlChhKlWq5PC79fTp0yY0NNS4u7ubvn37mkmTJplXXnnFNGnSxGzdutUYY0xqaqpp166dKVy4sBk4cKCZMmWK6d+/v/Hw8DD33HPPNeP9448/jCTTu3fvLL0XZ8+eNbVr1zaFChUygwYNMu+995657bbbjCQzYcIEez9n5IOVKlUyNWrUMMWLFzfDhg0zb7/9tqlbt65xc3Oz5ynGGLN8+XIjySxfvtzetnTpUuPp6WnCw8PNW2+9Zd555x1Tr1494+npadavX2/v98YbbxhJZt68ecYYY5KTk01wcLAJCQkx58+ft/dr1aqVadWqlTHGmISEBPPyyy8bSebxxx+350779+83e/fuNZLM+++/73AuKSkppkSJEll+n4GCjKIUgAxupCjl7+9vTp06ZW8fPny4kWTq169vLl68aG/v1q2b8fT0tP/iP336tClevLjp27evw+skJCQYPz+/DO3/VqJECVO/fv0sndeRI0eMp6enadeunUlNTbW3T5w40Ugyn376qb2tVatWRpKZOXOmve333383koybm5tZt26dvX3hwoVGkpk2bZq9Lb0odffddzvE8NRTTxlJZvv27fa2fxelXnnlFVOkSBHzxx9/OOw7bNgw4+7ubg4ePGhvmzJlipFkvvjiC7Nu3Trj7u5uBg4c6LDfv4tSxhhTp04de7J1pfTj7d6929524cIFU7p0aYcYAQDIrxYtWmTc3d2Nu7u7CQ8PN0OHDjULFy50+PIqnSQjyWzatMneduDAAePt7e3wpdHZs2cz7BsbG2skmenTp9vbRo4caSSZ7777LkP/tLQ0Y4wxn3/+uXFzczOrV6922D558mQjyaxZs+aq5zZv3jwjybzzzjtXfwOuMGHCBHueke7ChQsmPDzcFC1a1P6F3M3mg8ZczockmW+//dbelpiYaMqWLWvCwsLsbf8uSqWlpZnq1aubiIgI+3tkzOX3vEqVKqZt27b2ttTUVNOiRQtTpkwZc+zYMdOvXz/j4eFhNm7c6HDeVxaljDFm48aNGXK9dOHh4aZp06YObd99912GwhmAzHH7HgCnePDBB+Xn52dfb9q0qaTL81V5eHg4tF+4cEF///23JGnx4sU6deqUunXrpmPHjtkXd3d3NW3aVMuXL7/m6yYlJalYsWJZinHJkiW6cOGCBg4cKDe3/3389e3bV76+vvrxxx8d+hctWlQPP/ywfb1mzZoqXry4ateubT+/K8/1zz//zPCa/fr1c1h/+umnJUk//fTTVeOcPXu2brvtNpUoUcLhPWnTpo1SU1Mdbgl4/PHHFRERoaefflqPPvqogoOD9dprr2Xl7cjUQw89JG9vb82YMcPetnDhQh07dixPzDEGAMDNatu2rWJjY3X33Xdr+/btev311xUREaHy5cvr+++/z9A/PDzc/pAVSapYsaLuueceLVy4UKmpqZIkHx8f+/aLFy/q+PHjqlatmooXL64tW7bYt3377beqX7++7r333gyvk/6wktmzZ6t27dqqVauWQ55wxx13SNI1c6ekpCRJynLu9NNPPykwMFDdunWztxUqVEjPPPOMkpOTtXLlSof+N5oPpitXrpzDufv6+ioyMlJbt25VQkJCpjFu27ZNe/fu1SOPPKLjx4/b348zZ87ozjvv1KpVq+zTH7i5uSkmJkbJycnq0KGDPvzwQw0fPlyNGzfO0vuRmcjISK1fv1779++3t82YMUNBQUFq1arVDR8XKCgoSgFwiooVKzqspyckQUFBmbafPHlSkrR3715J0h133CF/f3+HZdGiRdedUNTX11enT5/OUowHDhyQdLm4dCVPT09VrVrVvj1dhQoV7AnglfFf75yuVL16dYf14OBgubm5ZTqHRLq9e/dqwYIFGd6PNm3aSMo4yeonn3yis2fPau/evYqJiXFIfLOrePHi6ty5s2bOnGlvmzFjhsqXL29PdgEUDKtWrVLnzp1Vrlw52Ww2zZ07N9vHMMbozTffVI0aNeTl5aXy5ctrzJgxzg8WcLImTZrou+++08mTJ7VhwwYNHz5cp0+f1gMPPKBdu3Y59P3373pJqlGjhs6ePWufa+jcuXMaOXKkfa7I0qVLy9/fX6dOnVJiYqJ9v/379ys0NPSase3du1e//fZbhjyhRo0akq49Gbuvr68kZSt3ql69usOXeZJUu3Zt+/Yr3Wg+mK5atWoZcq/087pa7pSeS0ZFRWV4Tz7++GOlpKQ4vMfBwcEaPXq0Nm7cqDp16mjEiBGZn3wWde3aVV5eXvYv9BITEzV//nx17949w7kAyIin7wFwiqs93e1q7cYYSbJ/c/X5558rMDAwQ78rv1XLTK1atbRt2zZduHDB6U9KudFzupasJCdpaWlq27athg4dmun29OQs3YoVK+wTm+7YsUPh4eHXfY1riYyM1OzZs7V27VrVrVtX33//vZ566qkMCSmA/O3MmTOqX7++evfurfvuu++GjjFgwAAtWrRIb775purWrasTJ05k+rALILfy9PRUkyZN1KRJE9WoUUO9evXS7NmzNWrUqGwd5+mnn9a0adM0cOBAhYeHy8/PTzabTQ8//LA9F8qqtLQ01a1bV2+//Xam2/9dALpS+kNhduzYka3XzKqcyJ2uJ/39e+ONN9SgQYNM+xQtWtRhfdGiRZKkf/75R8ePH880B82qEiVKqFOnTpoxY4ZGjhypb775RikpKYwwB7KIohQAlwoODpZ0+Wk26SOBsqNz586KjY3Vt99+6zC0PDOVKlWSJO3Zs0dVq1a1t1+4cEFxcXE39PrXs3fvXlWpUsW+vm/fPqWlpTk8wfDfgoODlZycnKV44uPj9fTTT6tdu3by9PTUkCFDFBERYT/Xq7lWcax9+/by9/fXjBkz1LRpU509e1aPPvrodWMBkL906NBBHTp0uOr2lJQUvfjii/ryyy916tQphYaGavz48fYnVu3evVuTJk3Szp077SNUr/w8BPKa9Fu84uPjHdrTR+pc6Y8//lDhwoXl7+8vSfrmm28UFRWlt956y97n/PnzGZ7+GxwcrJ07d14zjuDgYG3fvl133nlntkfi1KhRQzVr1tS8efP07rvvZijW/FulSpX066+/Ki0tzeHLqd9//92+3Zn27dsnY4zDef3xxx+SdNXcKT2X9PX1zVLuNHnyZC1evFhjxozR2LFj9Z///Efz5s275j7Xe58jIyN1zz33aOPGjZoxY4bCwsJUp06d68YCgNv3ALhYRESEfH199dprr+nixYsZtmf2iN0rPfHEEypbtqyeffZZe9JypSNHjujVV1+VJLVp00aenp567733HL6Z++STT5SYmKiOHTve5Nlk9MEHHzisv//++5J0zT/0HnroIcXGxmrhwoUZtp06dUqXLl2yr/ft21dpaWn65JNP9NFHH8nDw0N9+vS57jePRYoUyZAIp/Pw8FC3bt309ddfKyYmRnXr1lW9evWueTwABU///v0VGxurWbNm6ddff9WDDz6o9u3b2/9A/+GHH1S1alXNnz9fVapUUeXKlfXYY48xUgq53vLlyzP9PZo+H+S/pwGIjY11mBfq0KFDmjdvntq1a2cfIeTu7p7hmO+//759zql0999/v7Zv3645c+ZkeP30/R966CH9/fffmjp1aoY+586d05kzZ655ftHR0Tp+/Lgee+wxh5wi3aJFizR//nxJ0l133aWEhAR99dVX9u2XLl3S+++/r6JFizp9zqR//vnH4dyTkpI0ffp0NWjQ4KqjmRo1aqTg4GC9+eabSk5OzrD9ylwyLi5Ozz33nO6//3698MILevPNN/X9999r+vTp14yrSJEiknTV3KlDhw4qXbq0xo8fr5UrVzJKCsgGRkoBcClfX19NmjRJjz76qBo2bKiHH35Y/v7+OnjwoH788Uc1b95cEydOvOr+JUqU0Jw5c3TXXXepQYMG6tGjh32y0S1btujLL7+0387m7++v4cOHKzo6Wu3bt9fdd9+tPXv26MMPP1STJk1yJIGIi4vT3Xffrfbt2ys2NlZffPGFHnnkEdWvX/+q+zz33HP6/vvv1alTJ/Xs2VONGjXSmTNntGPHDn3zzTf666+/VLp0aU2bNk0//vijYmJiVKFCBUmXE9wePXpo0qRJeuqpp676Go0aNdKkSZP06quvqlq1agoICHCYMyoyMlLvvfeeli9frvHjxzvvDQGQLxw8eFDTpk3TwYMHVa5cOUnSkCFDtGDBAk2bNk2vvfaa/vzzTx04cECzZ8/W9OnTlZqaqkGDBumBBx7QsmXLXHwGwNU9/fTTOnv2rO69917VqlVLFy5c0Nq1a/XVV1+pcuXK6tWrl0P/0NBQRURE6JlnnpGXl5c+/PBDSZeLP+k6deqkzz//XH5+fgoJCVFsbKyWLFmiUqVKORzrueee0zfffKMHH3xQvXv3VqNGjXTixAl9//33mjx5surXr69HH31UX3/9tZ544gktX75czZs3V2pqqn7//Xd9/fXXWrhw4TUn7u7atat27NihMWPGaOvWrerWrZsqVaqk48ePa8GCBVq6dKl9bsnHH39cU6ZMUc+ePbV582ZVrlxZ33zzjdasWaMJEyZkecL0rKpRo4b69OmjjRs3qkyZMvr00091+PBhTZs27ar7uLm56eOPP1aHDh1Up04d9erVS+XLl9fff/+t5cuXy9fXVz/88IOMMerdu7d8fHw0adIkSdJ//vMfffvttxowYIDatGlj/zz7t+DgYBUvXlyTJ09WsWLFVKRIETVt2tQ++rNQoUJ6+OGHNXHiRLm7u1939D6AK7joqX8AcrF+/fqZq308REVFmUqVKtnX0x8B/MYbbzj0S39c7+zZsx3ap02bZiRlePTu8uXLTUREhPHz8zPe3t4mODjY9OzZ0+ERy9fyzz//mEGDBpkaNWoYb29vU7hwYdOoUSMzZswYk5iY6NB34sSJplatWqZQoUKmTJky5sknnzQnT5506NOqVStTp06dDK9TqVIl07Fjxwztkky/fv3s66NGjTKSzK5du8wDDzxgihUrZkqUKGH69+9vzp07l+GYUVFRDm2nT582w4cPN9WqVTOenp6mdOnSplmzZubNN980Fy5cMIcOHTJ+fn6mc+fOGWK59957TZEiRcyff/5pjPnfex4XF2fvk5CQYDp27GiKFStmJDk89jhdnTp1jJubm/nvf/+bYRuAgkWSmTNnjn19/vz5RpIpUqSIw+Lh4WEeeughY4wxffv2NZLMnj177Ptt3rzZSDK///671acAZNnPP/9sevfubWrVqmWKFi1qPD09TbVq1czTTz9tDh8+7NA3/ff/F198YapXr268vLxMWFiYWb58uUO/kydPml69epnSpUubokWLmoiICPP7779nmgMcP37c9O/f35QvX954enqaChUqmKioKHPs2DF7nwsXLpjx48ebOnXqGC8vL1OiRAnTqFEjEx0dnSHvuZqlS5eae+65xwQEBBgPDw/j7+9vOnfubObNm+fQ7/Dhw/bYPT09Td26dc20adMc+jgjH0zPsRYuXGjq1atnvLy8TK1atTLsm37Mf7/HW7duNffdd58pVaqU8fLyMpUqVTIPPfSQWbp0qTHGmHfffddIMt9++63DfgcPHjS+vr7mrrvusre1atUqQ240b948ExISYjw8PIykDO/Bhg0bjCTTrl07AyDrbMY4YXY5AICD0aNHKzo6WkePHlXp0qVdHc4NCQsLU8mSJbV06VJXhwLAxWw2m+bMmaMuXbpIkr766it1795dv/32W4YJjIsWLarAwECN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ZlqUGj58uAYPHmxfT0pKuuoImZtFnvQ/5EnkSQDgKrktT+L2PSc6cvrqidaN9LtRvXv3VkxMjD777LOrfnvWuXNnnThxQlOnTtX69eu1fv16Sf+bkPNarvwFfy1r166VJJ04cUInTpzIYvRShw4ddO7cOfXp00edO3dWqVIZE1M3N7cMj7fODXMA+Pj4XHN7bGysunfvrrvuukvz58/X1q1b9eKLL2bpfb8RhQoVcli32Wz2xBoA8iNX3x7mamfPnpWbm2N65+7uftXPfi8vL/n6+josOYU8yRF5UkbkSQCQs3JjnkRRyokCink7td+Nat++vS5cuKCLFy8qIiIiw/bjx49rz549eumll3TnnXeqdu3aOnnypEOf9G/4UlNTbyiG/fv3a9CgQZo6daqaNm2qqKioLP+S9/DwUGRkpFasWHHVZNHf318JCQkOCdf1Hsvs6emZpfO5dOmSNm3aZF/fs2ePTp06pdq1a1933+rVq8vHx0dLly7NdPvatWtVqVIlvfjii2rcuLGqV69uv4UgO3HWrl1b27dv15kzZ+xta9askZubm31SVQAoaK53e5h0+fYwK5705iqdO3fWmDFj9OOPP+qvv/7SnDlz9Pbbb+vee+91dWjkSVcgTyJPAgCr5dY8iaKUE91SpaTK+nnrandi2nR5WNwtVUrmaBzu7u7avXu3du3a5TAkOV2JEiVUqlQpffTRR9q3b5+WLVvmMHRfujxngY+PjxYsWKDDhw8rMTExy6+fmpqqHj16KCIiQr169dK0adP066+/6q233sryMV555RUdPXo002RRujxJ6NGjR/X6669r//79+uCDD/Tzzz9f85iVK1fW+vXr9ddffzkMxf+3QoUK6emnn9b69eu1efNm9ezZU7feeut1h6RLkre3t55//nkNHTpU06dP1/79+7Vu3Tp98sknki4nYwcPHtSsWbO0f/9+vffeexke7Vy5cmXFxcVp27ZtOnbsmFJSUjK8Tvfu3eXt7a2oqCjt3LlTy5cv19NPP61HH330upOkAkB+lZ3bw/Kr999/Xw888ICeeuop1a5dW0OGDNF//vMfvfLKK64OjTzp/5EnkScBgCvk1jyJopQTubvZNKpziCRlSLjS10d1DrFkArFrDcF3c3PTrFmztHnzZoWGhmrQoEF64403HPp4eHjovffe05QpU1SuXDndc889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sEKFCmV4ukrDhg21Z88eVatWLdN9atasqUOHDunw4cMqU+byeVw5WWb6Mb766isFBATI19fX6XGHhIRIujwkPTg4WJ6enlqzZo0qVaok6fJcDRs3btTAgQOveRxPT0+eLgMAgFXIk+zHIE8CAOAy5pRypkrNJN9ykmxX6WCTfMtf7pcLVK5cWUuXLlVCQoL90cEjR47U9OnTFR0drd9++027d+/WrFmz9NJLL0mSfa6BqKgo/frrr1qzZo19m812+by7d++u0qVL65577tHq1asVFxenFStW6JlnntF///vfbMX4wAMP6J133tH69et14MABrVixQv369VONGjVUq1YtFSlSRE8++aSee+45LViwQLt27VLfvn119uxZ9enT57rnHxcXp23btunYsWNKSUnJ7lsIAACyijxJEnkSAABXoijlTG7uUvvx/7/y74Tr/9fbj7vcLxd46623tHjxYgUFBSksLEySFBERofnz52vRokVq0qSJbr31Vr3zzjv2b9fc3d01d+5cJScnq0mTJnrsscfsw8PTh60XLlxYq1atUsWKFXXfffepdu3a6tOnj86fP5/tbwQjIiL0ww8/qHPnzqpRo4aioqJUq1YtLVq0SB4elwf6jRs3Tvfff78effRRNWzYUPv27dPChQtVokSJax77/vvvV/v27XX77bfL399fX375ZbZiAwAA2UCeJIk8CQCAK9mMMZk9AiXfSkpKkp+fnxITEzP84j9//rzi4uJUpUqVLM8LkKld319+usyVk3n6lr+caIXcfePHzaXWrFmjFi1aaN++fQoODnZ1OLhBTvv5B4AC4lo5RV5FnuR85En5A3kSAGRPVvMk5pTKCSF3S7U6Xn56TPLhy3MjVGqWa775u1lz5sxR0aJFVb16de3bt08DBgxQ8+bNSbQAAMD1kScBAID/R1Eqp7i5W/I4Y1c4ffq0nn/+eR08eFClS5dWmzZt9NZbb7k6LAAAkFeQJwEAAFGUwg2IjIxUZGSkq8MAAADIdciTAADIOiY6BwAAAAAAgOUoSgEAAAAAAMByFKUykZaW5uoQAMvxcw8AyAp+X6Ag4uceAHIGc0pdwdPTU25ubvrnn3/k7+8vT09P2Ww2V4cF5ChjjC5cuKCjR4/Kzc1Nnp6erg4JAJALkSehICJPAoCcRVHqCm5ubqpSpYri4+P1zz//uDocwFKF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5kMAAACAoy5duuTYsZOTk9W9e3dNnTpVr776ao69DgAAwM3KdlHq22+/1aOPPqru3btry5YtSklJkXT56XuvvfaafvrpJ6cHCQAAkJ+MGjUqx47dr18/dezYUW3atKEoBQAAcrVszyn16quvavLkyZo6daoKFSpkb2/evLm2bNni1OAAAADyo2w+/DjLZs2apS1btmjs2LFZ6p+SkqKkpCSHBQAAwCrZLkrt2bNHLVu2zNDu5+enU6dOOSMmAACAfK1OnTqaNWuWLly4cM1+e/fu1ZNPPqlx48Zd95iHDh3SgAEDNGPGDHl7e2cpjrFjx8rPz8++BAUFZWk/AAAAZ8j27XuBgYHat2+fKleu7ND+yy+/qGrVqs6KCwAAIN96//339fzzz+upp55S27Zt1bhxY5UrV07e3t46efKkdu3apV9++UW//fab+vfvryeffPK6x9y8ebOOHDmihg0b2ttSU1O1atUqTZw4USkpKXJ3d3fYZ/jw4Ro8eLB9PSkpicIUAACwTLaLUn379tWAAQP06aefymaz6Z9//lFsbKyGDBlyU48tBgAAKCjuvPNObdq0Sb/88ou++uorzZgxQwcOHNC5c+dUunRphYWFKTIyUt27d1eJEiWyfMwdO3Y4tPXq1Uu1atXS888/n6EgJUleXl7y8vJyyjkBAABkV7aLUsOGDVNaWpruvPNOnT17Vi1btpSXl5eGDBmip59+OidiBAAAyJdatGjhtCcXFytWTKGhoQ5tRYoUUalSpTK0AwAA5AbZLkrZbDa9+OKLeu6557Rv3z4lJycrJCRERYsWzYn4AAAAAAAAkA9luyiVztPTUyEhIUpKStKSJUtUs2ZN1a5d25mxAQAA4CasWLHC1SEAAABcVbafvvfQQw9p4sSJkqRz586pSZMmeuihh1SvXj19++23Tg8QAAAAAAAA+U+2i1KrVq3SbbfdJkmaM2eO0tLSdOrUKb333nt69dVXnR4gAAAAAAAA8p9sF6USExNVsmRJSdKCBQt0//33q3DhwurYsaP27t3r9AABAAAAAACQ/2S7KBUUFKTY2FidOXNGCxYsULt27SRJJ0+elLe3t9MDBAAAyO/279+vl156Sd26ddORI0ckST///LN+++03F0cGAACQc7JdlBo4cKC6d++uChUqqFy5cmrdurWky7f11a1b19nxAQAA5GsrV65U3bp1tX79en333XdKTk6WJG3fvl2jRo1ycXQAAAA5J9tFqaeeekrr1q3Tp59+ql9++UVubpcPUbVqVeaUAgAAyKZhw4bp1Vdf1eLFi+Xp6Wlvv+OOO7Ru3ToXRgYAAJCzPG5kp0aNGqlRo0YObR07dnRKQAAAAAXJjh07NHPmzAztAQEBOnbsmAsiAgAAsEaWRkqNGzdO586dy9IB169frx9//PGmggIAACgoihcvrvj4+AztW7duVfny5V0QEQAAgDWyVJTatWuXKlasqKeeeko///yzjh49at926dIl/frrr/rwww/VrFkzde3aVcWKFcuxgAEAAPKThx9+WM8//7wSEhJks9mUlpamNWvWaMiQIYqMjHR1eAAAADkmS0Wp6dOna8mSJbp48aIeeeQRBQYGytPTU8WKFZOXl5fCwsL06aefKjIyUr///rtatmyZ03EDAADkC6+99ppq1aqloKAgJScnKyQkRC1btlSzZs300ksvuTo8AACAHGMzxpjs7JCWlqZff/1VBw4c0Llz51S6dGk1aNBApUuXzqkYnSopKUl+fn5KTEyUr6+vq8MBAAB5lLNzikOHDmnHjh1KTk5WWFiYqlev7oQos4c8CQAAOENWc4psT3Tu5uamBg0aqEGDBjcTHwAAAK4QFBSkoKAgV4cBAABgmSzdvgcAAICccf/992v8+PEZ2l9//XU9+OCDLogIAADAGhSlAAAAXGjVqlW66667MrR36NBBq1atckFEAAAA1qAoBQAA4ELJycny9PTM0F6oUCElJSW5ICIAAABrUJQCAABwobp16+qrr77K0D5r1iyFhIS4ICIAAABrZHui83T79u3T/v371bJlS/n4+MgYI5vN5szYAAAA8r0RI0bovvvu0/79+3XHHXdIkpYuXaovv/xSs2fPdnF0AAAAOSfbRanjx4+ra9euWrZsmWw2m/bu3auqVauqT58+KlGihN56662ciBMAACBf6ty5s+bOnavXXntN33zzjXx8fFSvXj0tWbJErVq1cnV4AAAAOSbbt+8NGjRIHh4eOnjwoAoXLmxv79q1qxYsWODU4AAAAAqCjh07as2aNTpz5oyOHTumZcuWUZACAAD5XrZHSi1atEgLFy5UhQoVHNqrV6+uAwcOOC0wAAAAAAAA5F/ZLkqdOXPGYYRUuhMnTsjLy8spQQEAABQUqampeuedd/T111/r4MGDunDhgsP2EydOuCgyAACAnJXt2/duu+02TZ8+3b5us9mUlpam119/XbfffrtTgwMAAMjvoqOj9fbbb6tr165KTEzU4MGDdd9998nNzU2jR492dXgAAAA5JtsjpV5//XXdeeed2rRpky5cuKChQ4fqt99+04kTJ7RmzZqciBEAACDfmjFjhqZOnaqOHTtq9OjR6tatm4KDg1WvXj2tW7dOzzzzjKtDBAAAyBHZHikVGhqqP/74Qy1atNA999yjM2fO6L777tPWrVsVHBycrWOtWrVKnTt3Vrly5WSz2TR37tzr7rNixQo1bNhQXl5eqlatmmJiYrJ7CgAAALlGQkKC6tatK0kqWrSoEhMTJUmdOnXSjz/+6MrQAAAAclS2R0pJkp+fn1588cWbfvEzZ86ofv366t27t+67777r9o+Li1PHjh31xBNPaMaMGVq6dKkee+wxlS1bVhERETcdDwAAgNUqVKig+Ph4VaxYUcHBwVq0aJEaNmyojRs3Ml8nAADI126oKHX+/Hn9+uuvOnLkiNLS0hy23X333Vk+TocOHdShQ4cs9588ebKqVKmit956S5JUu3Zt/fLLL3rnnXcoSgEAgDzp3nvv1dKlS9W0aVM9/fTT6tGjhz755BMdPHhQgwYNcnV4AAAAOSbbRakFCxYoMjJSx44dy7DNZrMpNTXVKYFlJjY2Vm3atHFoi4iI0MCBA6+6T0pKilJSUuzrSUlJORUeAABAto0bN87+765du6pixYqKjY1V9erV1blzZxdGBgAAkLOyPafU008/rQcffFDx8fFKS0tzWHKyICVdnnOhTJkyDm1lypRRUlKSzp07l+k+Y8eOlZ+fn30JCgrK0RgBAABuRnh4uAYPHkxBCgAA5HvZLkodPnxYgwcPzlAcyq2GDx+uxMRE+3Lo0CFXhwQAAOBgz5496t+/v+68807deeed6t+/v/bs2ePqsAAAAHJUtotSDzzwgFasWJEDoVxfYGCgDh8+7NB2+PBh+fr6ysfHJ9N9vLy85Ovr67AAAADkFt9++61CQ0O1efNm1a9fX/Xr19eWLVsUGhqqb7/91tXhAQAA5Jhszyk1ceJEPfjgg1q9erXq1q2rQoUKOWx/5plnnBbcv4WHh+unn35yaFu8eLHCw8Nz7DUBAABy0tChQzV8+HC9/PLLDu2jRo3S0KFDdf/997soMgAAgJyV7aLUl19+qUWLFsnb21srVqyQzWazb7PZbNkqSiUnJ2vfvn329bi4OG3btk0lS5ZUxYoVNXz4cP3999+aPn26JOmJJ57QxIkTNXToUPXu3VvLli3T119/rR9//DG7pwEAAJArxMfHKzIyMkN7jx499MYbb7ggIgAAAGtkuyj14osvKjo6WsOGDZObW7bv/nOwadMm3X777fb1wYMHS5KioqIUExOj+Ph4HTx40L69SpUq+vHHHzVo0CC9++67qlChgj7++GNFRETcVBwAAACu0rp1a61evVrVqlVzaP/ll1902223uSgqAACAnGczxpjs7FCyZElt3LhRwcHBORVTjkpKSpKfn58SExOZXwoAANwwZ+UUkydP1siRI/XQQw/p1ltvlSStW7dOs2fPVnR0tMqVK2fve/fdd9903NdCngQAAJwhqzlFtotSgwYNkr+/v1544YWbDtIVSLYAAIAzOCunyOrIc5vNptTU1Bt+nawgTwIAAM6Q1Zwi27fvpaam6vXXX9fChQtVr169DBOdv/3229mPFgAAoIBKS0tzdQgAAAAuke2i1I4dOxQWFiZJ2rlzp8O2Kyc9BwAAAAAAAK4m20Wp5cuX50QcAAAABUpsbKyOHz+uTp062dumT5+uUaNG6cyZM+rSpYvef/99eXl5uTBKAACAnHNzj88DAADADXn55Zf122+/2dd37NihPn36qE2bNho2bJh++OEHjR071oURAgAA5KwsjZS67777FBMTI19fX913333X7Pvdd985JTAAAID8bNu2bXrllVfs67NmzVLTpk01depUSVJQUJBGjRql0aNHuyhCAACAnJWlopSfn599vig/P78cDQgAAKAgOHnypMqUKWNfX7lypTp06GBfb9KkiQ4dOuSK0AAAACyRpaLUtGnT9PLLL2vIkCGaNm1aTscEAACQ75UpU0ZxcXEKCgrShQsXtGXLFkVHR9u3nz59OsNTjgEAAPKTLM8pFR0dreTk5JyMBQAAoMC46667NGzYMK1evVrDhw9X4cKFddttt9m3//rrrwoODnZhhAAAADkry0/fM8bkZBwAAAAFyiuvvKL77rtPrVq1UtGiRfXZZ5/J09PTvv3TTz9Vu3btXBghAABAzspyUUqSfV4pAAAA3JzSpUtr1apVSkxMVNGiReXu7u6wffbs2SpatKiLogMAAMh52SpK1ahR47qFqRMnTtxUQAAAAAXJ1R4iU7JkSYsjAQAAsFa2ilLR0dE8fQ8AAAAAAAA3LVtFqYcfflgBAQE5FQsAAABuwqRJkzRp0iT99ddfkqQ6depo5MiR6tChg2sDAwAAyESWn77HfFIAAAC5W4UKFTRu3Dht3rxZmzZt0h133KF77rlHv/32m6tDAwAAyICn7wEAAOQTnTt3dlgfM2aMJk2apHXr1qlOnTouigoAACBzWS5KpaWl5WQcAAAAcKLU1FTNnj1bZ86cUXh4uKvDAQAAyCBbc0oBAAAgd9uxY4fCw8N1/vx5FS1aVHPmzFFISEimfVNSUpSSkmJfT0pKsipMAACArM8pBQAAgNyvZs2a2rZtm9avX68nn3xSUVFR2rVrV6Z9x44dKz8/P/sSFBRkcbQAAKAgs5kCNllUUlKS/Pz8lJiYKF9fX1eHAwAA8qi8klO0adNGwcHBmjJlSoZtmY2UCgoKyvXnBAAAcres5kncvgcAAJCPpaWlORSeruTl5SUvLy+LIwIAALiMohQAAEA+MXz4cHXo0EEVK1bU6dOnNXPmTK1YsUILFy50dWgAAAAZUJQCAADIJ44cOaLIyEjFx8fLz89P9erV08KFC9W2bVtXhwYAAJABRSkAAIB84pNPPnF1CAAAAFnG0/cAAAAAAABgOYpSAAAAAAAAsBxFKQAAAAAAAFiOohQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlAIAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAAAAAAABYjqIUAAAAAAAALEdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACW83B1AACQFalpRhviTujI6fMKKOatW6qUlLubzdVhAQAAAABuEEUpALnegp3xiv5hl+ITz9vbyvp5a1TnELUPLevCyAAAAAAAN4rb9wDkagt2xuvJL7Y4FKQkKSHxvJ78YosW7Ix3UWQAAAAAgJtBUQpArpWaZhT9wy6ZTLalt0X/sEupaZn1AAAAAADkZhSlAORaG+JOZBghdSUjKT7xvDbEnbAuKAAAAACAU1CUApBrHTl99YLUjfQDAAAAAOQeFKUA5FoBxbyd2g8AAAAAkHtQlAKQa91SpaTK+nnLdpXtNl1+Ct8tVUpaGRYAAAAAwAkoSgHItdzdbBrVOUSSMhSm0tdHdQ6Ru9vVylYAAAAAgNyKohSAXK19aFlN6tFQgX6Ot+gF+nlrUo+Gah9a1kWRAQAAAABuhoerAwCA62kfWlZtQwK1Ie6Ejpw+r4Bil2/ZY4QUAAAAAORdFKUA5AnubjaFB5dydRgAAAAAACfh9j0AAAAAAABYjqIUAAAAAAAALEdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACWoygFAAAAAAAAy1GUAgAAAAAAgOUoSgEAAAAAAMByFKUAAADyibFjx6pJkyYqVqyYAgIC1KVLF+3Zs8fVYQEAAGSKohQAAEA+sXLlSvXr10/r1q3T4sWLdfHiRbVr105nzpxxdWgAAAAZeLg6AAAAADjHggULHNZjYmIUEBCgzZs3q2XLli6KCgAAIHOMlAIAAMinEhMTJUklS5Z0cSQAAAAZMVIKAAAgH0pLS9PAgQPVvHlzhYaGZtonJSVFKSkp9vWkpCSrwgMAAGCkFAAAQH7Ur18/7dy5U7Nmzbpqn7Fjx8rPz8++BAUFWRghAAAo6ChKAQAA5DP9+/fX/PnztXz5clWoUOGq/YYPH67ExET7cujQIQujBOAsqWlGsfuPa962vxW7/7hS04yrQwKALOH2PQAAgHzCGKOnn35ac+bM0YoVK1SlSpVr9vfy8pKXl5dF0QHICQt2xiv6h12KTzxvbyvr561RnUPUPrSsCyMDgOtjpBQAAEA+0a9fP33xxReaOXOmihUrpoSEBCUkJOjcuXOuDg1ADliwM15PfrHFoSAlSQmJ5/XkF1u0YGe8iyIDgKyhKAUAAJBPTJo0SYmJiWrdurXKli1rX7766itXhwbAyVLTjKJ/2KXMbtRLb4v+YRe38gHI1bh9DwAAIJ8whj8+gYJiQ9yJDCOkrmQkxSee14a4EwoPLmVdYACQDYyUAgAAAIA85sjpqxekbqQfALgCRSkAAAAAyGMCink7tR8AuAJFKQAAAADIY26pUlJl/bxlu8p2my4/he+WKiWtDAsAsoWiFAAAAADkMe5uNo3qHCJJGQpT6eujOofI3e1qZSsAcD2KUgAAAACQB7UPLatJPRoq0M/xFr1AP29N6tFQ7UPLuigyAMganr4HAAAAAHlU+9CyahsSqA1xJ3Tk9HkFFLt8yx4jpADkBblipNQHH3ygypUry9vbW02bNtWGDRuu2jcmJkY2m81h8fZm8j4AAAAABZO7m03hwaV0T4PyCg8uRUEKQJ7h8qLUV199pcGDB2vUqFHasmWL6tevr4iICB05cuSq+/j6+io+Pt6+HDhwwMKIAQAAAAAAcLNcXpR6++231bdvX/Xq1UshISGaPHmyChcurE8//fSq+9hsNgUGBtqXMmXKWBgxAAAAAAAAbpZLi1IXLlzQ5s2b1aZNG3ubm5ub2rRpo9jY2Kvul5ycrEqVKikoKEj33HOPfvvtt6v2TUlJUVJSksMCAAAAAAAA13JpUerYsWNKTU3NMNKpTJkySkhIyHSfmjVr6tNPP9W8efP0xRdfKC0tTc2aNdN///vfTPuPHTtWfn5+9iUoKMjp5wEAAAAAAIDscfnte9kVHh6uyMhINWjQQK1atdJ3330nf39/TZkyJdP+w4cPV2Jion05dOiQxREDAAAAAADg3zxc+eKlS5eWu7u7Dh8+7NB++PBhBQYGZukYhQoVUlhYmPbt25fpdi8vL3l5ed10rAAAAAAAAHAel46U8vT0VKNGjbR06VJ7W1pampYuXarw8PAsHSM1NVU7duxQ2bJlcypMAAAAAAAAOJlLR0pJ0uDBgxUVFaXGjRvrlltu0YQJE3TmzBn16tVLkhQZGany5ctr7NixkqSXX35Zt956q6pVq6ZTp07pjTfe0IEDB/TYY4+58jQAAAAAAACQDS4vSnXt2lVHjx7VyJEjlZCQoAYNGmjBggX2yc8PHjwoN7f/Deg6efKk+vbtq4SEBJUoUUKNGjXS2rVrFRIS4qpTAAAAAAAAQDbZjDHG1UFYKSkpSX5+fkpMTJSvr6+rwwEAAHlUfswp8uM5AQAA62U1p8hzT98DAAAAAABA3kdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACWoygFAAAAAAAAy1GUAgAAAAAAgOUoSgEAAAAAAMByFKUAAAAAAABgOYpSAAAAAAAAsBxFKQAAAAAAAFiOohQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlAIAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAAAAAAABYjqIUAAAAAAAALEdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAD5xKpVq9S5c2eVK1dONptNc+fOdXVIAAAAV0VRCgAAIJ84c+aM6tevrw8++MDVoQAAAFyXh6sDAAAAgHN06NBBHTp0cHUYAAAAWcJIKQAAAAAAAFiOkVIAAAAFVEpKilJSUuzrSUlJLowGAAAUNIyUAgAAKKDGjh0rPz8/+xIUFOTqkAAAQAFCUQoAAKCAGj58uBITE+3LoUOHXB0SAAAoQLh9DwAAoIDy8vKSl5eXq8MAAAAFFEUpAACAfCI5OVn79u2zr8fFxWnbtm0qWbKkKlas6MLIAAAAMqIoBQAAkE9s2rRJt99+u3198ODBkqSoqCjFxMS4KCoAAIDMUZQCAADIJ1q3bi1jjKvDAAAAyBImOgcAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAAAAAAABYjqIUAAAAAAAALEdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACWoygFAAAAAAAAy1GUAgAAAAAAgOUoSgEAAAAAAMByFKUAAAAAAABgOYpSAAAAAAAAsBxFKQAAAAAAAFiOohQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlAIAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAAAAAAABYjqIUAAAAAAAALEdRCgAAAAAAAJajKAUAAAAAAADLUZQCAAAAAACA5ShKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACWoygFAAAAAAAAy1GUAgAAAAAAgOUoSgEAAAAAAMByHq4OAAAAwOnSUqUDa6Xkw1LRMlKlZpKbu6ujAgAAwBUoSgEAgPxl1/fSguelpH/+1+ZbTmo/Xgq523VxAQAAwAG37wEAgPxj1/fS15GOBSlJSoq/3L7re9fEBQAAgAwoSgHIG9JSpbjV0o5vLv83LdXVEQHIbdJSL4+Qkslk4/+3LRjG5wcAAEAuwe17TpSaZrQh7oSOnD6vgGLeuqVKSbm72VwdFpD3cSsOgKw4sDbjCCkHRkr6+3K/KrdZFhYAAAAylytGSn3wwQeqXLmyvL291bRpU23YsOGa/WfPnq1atWrJ29tbdevW1U8//WRRpFe3YGe8Wo5brHc/+VRLv/5Q737yqVqOW6wFO+NdHRqQt3ErDoCsSj7s3H55WHZzKyukXrqk39b8qE3zP9Jva35U6qVLrg4JyD8YUQ4gO3LRZ4bLR0p99dVXGjx4sCZPnqymTZtqwoQJioiI0J49exQQEJCh/9q1a9WtWzeNHTtWnTp10syZM9WlSxdt2bJFoaGhLjiDywWpuTMna3ah6SrnecLe/k9KSb08M1J65Am1Dy3rktiAPO26t+LYLt+KU6sjT9UCcPkpe87sl0dlN7eywtaFn6lcbLTq6Li97fDiUvonfJTCIqJcEhOQbzCiHEB25LLPDJsxJrO/9izTtGlTNWnSRBMnTpQkpaWlKSgoSE8//bSGDRuWoX/Xrl115swZzZ8/39526623qkGDBpo8efJ1Xy8pKUl+fn5KTEyUr6/vTcefmmb04muv6bWLr0uSrrxbL+3/39kXCg3VmBde4FY+ILviVkufdbp+v6j53IoD4HIhe0Lo5ZGUmRazbZeTroE7nFLIdnZO4SzZza2ulBPntHXhZ6q/9hlJmedJ25u9R2EKuFHpI8ozfOb9//9sD02nMAXgfyz8zMhqTuHS2/cuXLigzZs3q02bNvY2Nzc3tWnTRrGxsZnuExsb69BfkiIiIq7aP6dt2H9Uz1z8WJJjonXl+jMXP9GG/UctjgzIB7gVB0B2uLlf/pZPkj25svv/9fbj8vXIyhvJrXJS6qVLKhcbfTmOq+RJZWOjuZUPuBE83AFAduTSzwyXFqWOHTum1NRUlSnjOIy+TJkySkhIyHSfhISEbPVPSUlRUlKSw+JMqX+tUTnbiQyJVjo3m1TOdlypf61x6usCBQK34gDIrpC7L3/L5/uv2+Z9yxWIEQPZza1yOk/6ff1CldHxa+ZJgTqu39cvdOrrAgVCdh7uAAC59DPD5XNK5bSxY8cqOjo6x44fYDvl1H4ArlCp2eU/JK93K06lZlZHBiA3C7n78lxzB9ZeHklZtMzlz4l8PELqRuV0nnTu5N9O7QfgCowoB5AdufQzw6UjpUqXLi13d3cdPux40ocPH1ZgYGCm+wQGBmar//Dhw5WYmGhfDh065Jzg/19w1WCn9gNwBW7FAXCj3NwvzzVX94HL/y0gnxPZza1yOk/yKVHeqf0AXIER5QCyI5d+Zri0KOXp6alGjRpp6dKl9ra0tDQtXbpU4eHhme4THh7u0F+SFi9efNX+Xl5e8vX1dVicyb1yc53zCbRP1vlvaUY65xMo98rNnfq6QIFRwG/FAYDsyG5uldN5Uq2mETqsUtfMkxJUSrWaRjj1dYECIX1EeYYv7tLZJN/yjCgHcFku/cxwaVFKkgYPHqypU6fqs88+0+7du/Xkk0/qzJkz6tWrlyQpMjJSw4cPt/cfMGCAFixYoLfeeku///67Ro8erU2bNql///6uOQE3d/l0fkM2m01p/9qUJslms8mn8xsF5htaIEeE3C0N3Hn5KXv3f3L5vwN3UJACgExcL7eykruHh/4JHyVJGQpT6evx4aPk7pHvZ5QAnI8R5QCyI5d+Zrg8A+jatauOHj2qkSNHKiEhQQ0aNNCCBQvsE3QePHhQbm7/q501a9ZMM2fO1EsvvaQXXnhB1atX19y5cxUaGuqqU5BC7pbtoemXZ7K/YuIwm2952dqP4w9nwBnSb8UBAFzT9XIrq4VFRGmrpHKx0Sqj4/b2I7ZSig8fpbCIKJfEBeQL6SPK//V3iHzLXf7jkr9DAFwpF35m2IwxVxlQnT8lJSXJz89PiYmJTh+irrRUJlUFAKCAyNGcwkVy8pxSL13S7+sX6tzJv+VTorxqNY1ghBTgLPwdAiA7LPjMyGpOQSbgTIzkAAAAyJS7h4fqNO/o6jCA/Im/QwBkRy76zHD5nFIAAAAAAAAoeChKAQAAAAAAwHIUpQAAAAAAAGA5ilIAAAAAAACwHEUpAAAAAAAAWI6iFAAAAAAAACxHUQoAAAAAAACWoygFAAAAAAAAy1GUAgAAAAAAgOUoSgEAAAAAAMByFKUAAAAAAABgOYpSAAAAAAAAsBxFKQAAAAAAAFiOohQAAAAAAAAsR1EKAAAAAAAAlvNwdQBWM8ZIkpKSklwcCQAAyMvSc4n03CI/IE8CAADOkNU8qcAVpU6fPi1JCgoKcnEkAAAgPzh9+rT8/PxcHYZTkCcBAABnul6eZDP56eu9LEhLS9M///yjYsWKyWazuTqcPCMpKUlBQUE6dOiQfH19XR0OroPrlbdwvfIOrlXektPXyxij06dPq1y5cnJzyx8zIpAn3Rg+G/IWrlfewvXKO7hWeUtuyZMK3EgpNzc3VahQwdVh5Fm+vr58wOQhXK+8heuVd3Ct8pacvF75ZYRUOvKkm8NnQ97C9cpbuF55B9cqb3F1npQ/vtYDAAAAAABAnkJRCgAAAAAAAJajKIUs8fLy0qhRo+Tl5eXqUJAFXK+8heuVd3Ct8hauF6zCz1rewvXKW7heeQfXKm/JLderwE10DgAAAAAAANdjpBQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlCrAxo4dqyZNmqhYsWIKCAhQly5dtGfPHoc+58+fV79+/VSqVCkVLVpU999/vw4fPuzQ5+DBg+rYsaMKFy6sgIAAPffcc7p06ZKVp1IgjRs3TjabTQMHDrS3cb1yj7///ls9evRQqVKl5OPjo7p162rTpk327cYYjRw5UmXLlpWPj4/atGmjvXv3OhzjxIkT6t69u3x9fVW8eHH16dNHycnJVp9KvpeamqoRI0aoSpUq8vHxUXBwsF555RVdOeUi18t1Vq1apc6dO6tcuXKy2WyaO3euw3ZnXZtff/1Vt912m7y9vRUUFKTXX389p08NuRx5Ut5FjpT7kSflHeRJuVu+yJMMCqyIiAgzbdo0s3PnTrNt2zZz1113mYoVK5rk5GR7nyeeeMIEBQWZpUuXmk2bNplbb73VNGvWzL790qVLJjQ01LRp08Zs3brV/PTTT6Z06dJm+PDhrjilAmPDhg2mcuXKpl69embAgAH2dq5X7nDixAlTqVIl07NnT7N+/Xrz559/moULF5p9+/bZ+4wbN874+fmZuXPnmu3bt5u7777bVKlSxZw7d87ep3379qZ+/fpm3bp1ZvXq1aZatWqmW7durjilfG3MmDGmVKlSZv78+SYuLs7Mnj3bFC1a1Lz77rv2Plwv1/npp5/Miy++aL777jsjycyZM8dhuzOuTWJioilTpozp3r272blzp/nyyy+Nj4+PmTJlilWniVyIPClvIkfK/ciT8hbypNwtP+RJFKVgd+TIESPJrFy50hhjzKlTp0yhQoXM7Nmz7X12795tJJnY2FhjzOX/Cdzc3ExCQoK9z6RJk4yvr69JSUmx9gQKiNOnT5vq1aubxYsXm1atWtkTLq5X7vH888+bFi1aXHV7WlqaCQwMNG+88Ya97dSpU8bLy8t8+eWXxhhjdu3aZSSZjRs32vv8/PPPxmazmb///jvngi+AOnbsaHr37u3Qdt9995nu3bsbY7heucm/ky1nXZsPP/zQlChRwuFz8Pnnnzc1a9bM4TNCXkKelPuRI+UN5El5C3lS3pFX8yRu34NdYmKiJKlkyZKSpM2bN+vixYtq06aNvU+tWrVUsWJFxcbGSpJiY2NVt25dlSlTxt4nIiJCSUlJ+u233yyMvuDo16+fOnbs6HBdJK5XbvL999+rcePGevDBBxUQEKCwsDBNnTrVvj0uLk4JCQkO18rPz09NmzZ1uFbFixdX48aN7X3atGkjNzc3rV+/3rqTKQCaNWumpUuX6o8//pAkbd++Xb/88os6dOggieuVmznr2sTGxqply5by9PS094mIiNCePXt08uRJi84GuR15Uu5HjpQ3kCflLeRJeVdeyZM8bvoIyBfS0tI0cOBANW/eXKGhoZKkhIQEeXp6qnjx4g59y5Qpo4SEBHufK395p29P3wbnmjVrlrZs2aKNGzdm2Mb1yj3+/PNPTZo0SYMHD9YLL7ygjRs36plnnpGnp6eioqLs73Vm1+LKaxUQEOCw3cPDQyVLluRaOdmwYcOUlJSkWrVqyd3dXampqRozZoy6d+8uSVyvXMxZ1yYhIUFVqlTJcIz0bSVKlMiR+JF3kCflfuRIeQd5Ut5CnpR35ZU8iaIUJF3+Zmnnzp365ZdfXB0KruLQoUMaMGCAFi9eLG9vb1eHg2tIS0tT48aN9dprr0mSwsLCtHPnTk2ePFlRUVEujg7/9vXXX2vGjBmaOXOm6tSpo23btmngwIEqV64c1wuAJPKk3I4cKW8hT8pbyJOQ07h9D+rfv7/mz5+v5cuXq0KFCvb2wMBAXbhwQadOnXLof/jwYQUGBtr7/PvJJenr6X3gHJs3b9aRI0fUsGFDeXh4yMPDQytXrtR7770nDw8PlSlThuuVS5QtW1YhISEObbVr19bBgwcl/e+9zuxaXHmtjhw54rD90qVLOnHiBNfKyZ577jkNGzZMDz/8sOrWratHH31UgwYN0tixYyVxvXIzZ10bPhtxLeRJuR85Ut5CnpS3kCflXXklT6IoVYAZY9S/f3/NmTNHy5YtyzAkr1GjRipUqJCWLl1qb9uzZ48OHjyo8PBwSVJ4eLh27Njh8IO8ePFi+fr6Zvhlg5tz5513aseOHdq2bZt9ady4sbp3727/N9crd2jevHmGx4b/8ccfqlSpkiSpSpUqCgwMdLhWSUlJWr9+vcO1OnXqlDZv3mzvs2zZMqWlpalp06YWnEXBcfbsWbm5Of46dHd3V1pamiSuV27mrGsTHh6uVatW6eLFi/Y+ixcvVs2aNbl1rwAjT8o7yJHyFvKkvIU8Ke/KM3mSU6ZLR5705JNPGj8/P7NixQoTHx9vX86ePWvv88QTT5iKFSuaZcuWmU2bNpnw8HATHh5u357++Nx27dqZbdu2mQULFhh/f38en2uRK58sYwzXK7fYsGGD8fDwMGPGjDF79+41M2bMMIULFzZffPGFvc+4ceNM8eLFzbx588yvv/5q7rnnnkwfzxoWFmbWr19vfvnlF1O9enUenZsDoqKiTPny5e2POv7uu+9M6dKlzdChQ+19uF6uc/r0abN161azdetWI8m8/fbbZuvWrebAgQPGGOdcm1OnTpkyZcqYRx991OzcudPMmjXLFC5c2GmPOkbeRJ6Ut5Ej5V7kSXkLeVLulh/yJIpSBZikTJdp06bZ+5w7d8489dRTpkSJEqZw4cLm3nvvNfHx8Q7H+euvv0yHDh2Mj4+PKV26tHn22WfNxYsXLT6bgunfCRfXK/f44YcfTGhoqPHy8jK1atUyH330kcP2tLQ0M2LECFOmTBnj5eVl7rzzTrNnzx6HPsePHzfdunUzRYsWNb6+vqZXr17m9OnTVp5GgZCUlGQGDBhgKlasaLy9vU3VqlXNiy++6PDYW66X6yxfvjzT31VRUVHGGOddm+3bt5sWLVoYLy8vU758eTNu3DirThG5FHlS3kaOlLuRJ+Ud5Em5W37Ik2zGGHPz460AAAAAAACArGNOKQAAAAAAAFiOohQAAAAAAAAsR1EKAAAAAAAAlqMoBQAAAAAAAMtRlAIAAAAAAIDlKEoBAAAAAADAchSlAAAAAAAAYDmKUgAAAAAAALAcRSkAyAGtW7fWwIEDXR0GAABArkOeBCAdRSkAuU7Pnj3VpUsXy183JiZGxYsXv26/1NRUjRs3TrVq1ZKPj49Kliyppk2b6uOPP7b3+e677/TKK6/kYLQAAKAgIk8CkJ94uDoAAMhroqOjNWXKFE2cOFGNGzdWUlKSNm3apJMnT9r7lCxZ0oURAgAAuAZ5EoDsYKQUgFyvdevWeuaZZzR06FCVLFlSgYGBGj16tEMfm82mSZMmqUOHDvLx8VHVqlX1zTff2LevWLFCNptNp06dsrdt27ZNNptNf/31l1asWKFevXopMTFRNptNNpstw2uk+/777/XUU0/pwQcfVJUqVVS/fn316dNHQ4YMcYg5fVh6+mv/e+nZs6e9/7x589SwYUN5e3uratWqio6O1qVLl272rQMAAPkceRKAvIyiFIA84bPPPlORIkW0fv16vf7663r55Ze1ePFihz4jRozQ/fffr+3bt6t79+56+OGHtXv37iwdv1mzZpowYYJ8fX0VHx+v+Ph4h+TpSoGBgVq2bJmOHj2a5WOnHzM+Pl7Lli2Tt7e3WrZsKUlavXq1IiMjNWDAAO3atUtTpkxRTEyMxowZk6XjAwCAgo08CUBeRVEKQJ5Qr149jRo1StWrV1dkZKQaN/6/du4nFLo9juP4Z6ZnhNQk1JmFv2maLEY0Uwxb2ZgSIyvUNGqsWImUMCwoI8nCwkY0YmFBqcnsJNkpC2wsLIjtsNFwF3rcZ6LnunruuYb3q87id875/ekspm+f+Z3jUSKRSLuno6NDoVBITqdTkUhEHo9HCwsL7xo/KytLdrtdFotFhmHIMAzl5eW9eW80GtXt7a0Mw5Db7VY4HNbu7u5vx/45ps1mUygUUjAYVDAYlPS8zX1oaEg9PT2qqKhQU1OTIpGIlpaW3vl0AADAd0adBCBTEUoByAhutzut7XA4dHNzk3auvr7+Vfu9/wD+G1VVVTo5OdHh4aGCwaBubm7k9/sVCoV+2+/h4UHt7e0qLS3V/Pz8y/nj42NNTEwoLy/v5ejt7dXV1ZXu7+//+PoBAMDXQp0EIFPxoXMAGcFms6W1LRaLHh8f393fan3O4J+enl7OPTw8fHg9VqtVXq9XXq9XAwMDWl1dVVdXl0ZGRlReXv5mn76+Pl1eXuro6Eg/fvz985tMJjU+Pq62trZXfbKzsz+8RgAA8D1QJwHIVIRSAL6Mw8NDdXd3p7VramokSUVFRZKkq6sr5efnS3r+gOevsrKylEqlPjR3VVWVJOnu7u7N69FoVBsbGzo4OFBBQUHatdraWp2dnamysvJDcwMAAPwT6iQAnxGhFIAvY3NzUx6PR42NjVpbW9PR0ZGWl5clSZWVlSouLtbY2JimpqZ0fn6u2dnZtP5lZWVKJpNKJBKqrq5Wbm6ucnNzX80TCATU0NAgn88nwzB0cXGh4eFhOZ1OuVyuV/fv7e1pcHBQi4uLKiws1PX1tSQpJydHdrtdo6OjamlpUUlJiQKBgKxWq46Pj3VycqLJycn/4EkBAIDvhjoJwGfEN6UAfBnj4+NaX1+X2+3WysqKYrHYyz9zNptNsVhMp6encrvdmp6eflXI+Hw+hcNhdXZ2qqioSDMzM2/O09zcrO3tbfn9fjmdTvX09Mjlcikej6dtN/9pf39fqVRK4XBYDofj5ejv738Zb2dnR/F4XF6vV3V1dZqbm1NpaekffkIAAOC7ok4C8BlZnn59cRgAMpTFYtHW1pZaW1v/76UAAAB8KtRJAD4rdkoBAAAAAADAdIRSAAAAAAAAMB2v7wEAAAAAAMB07JQCAAAAAACA6QilAAAAAAAAYDpCKQAAAAAAAJiOUAoAAAAAAACmI5QCAAAAAACA6QilAAAAAAAAYDpCKQAAAAAAAJiOUAoAAAAAAACmI5QCAAAAAACA6f4Cuy4k/pWHrYwAAAAASUVORK5CYII=\n"
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
      "execution_count": 4,
      "outputs": []
    }
  ]
}