# vef_FinalProject_Heritage-Health-Prize
This is final project in course ML2020 at VEF

# . References:
<font color='blue' font-family= "Times New Roman">
<p>This project using data from <a href="https://foreverdata.org/1015/index.html">Forever Data</a> 
</p>
<p>
Feature Engineering is based on an idea <a href="https://foreverdata.org/1015/content/milestone1-2.pdf">Milestone 1-2 PDF</a> download here.</p>

    Project
    ├── 0.Data
    │ 
    ├── 1.Exploration
    │   ├── Explore_Raw Data.ipynb
    │   └── Explore_Feature after process.ipynb <──┐
    │                                              │ 
    ├── 2.Feature Engineering──────────────────────┘
    │   ├── 1_members processing.ipynb
    │   ├── 2_Claims processing.ipynb
    │   ├── 3_drugs_preprocess.ipynb
    │   ├── 4_LabCount_Preprocess.ipynb
    │   └── 5_merge_all.ipynb
    ├── 3.Modelling
    │   ├── 1_linear-regression.ipynb
    │   ├── 2_Logistic-Regression.ipynb
    │   ├── 3_random-forest-regression.ipynb
    │   ├── 4_Processing-Choice feature.ipynb
    │   ├── 5_Neuron_network.ipynb
    │   └── 6_gradient-boosting-classifier.ipynb
    └── 4.Evaluation
        ├── voting-model.ipynb ───> result.csv
        └── voting-model-newdata feature.ipynb ───> result_2.csv


# 1.Exploration:
This is step exploring raw data and feature data after processing in step 2.
## Expore raw data
This step tells us the data structure, the data form, some data visualization so we know what to do with this data before we put it into model training.

### Input data: Claims.csv 
shape(2668990, 14)

<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>MemberID</th>
      <th>ProviderID</th>
      <th>Vendor</th>
      <th>PCP</th>
      <th>Year</th>
      <th>Specialty</th>
      <th>PlaceSvc</th>
      <th>PayDelay</th>
      <th>LengthOfStay</th>
      <th>DSFS</th>
      <th>PrimaryConditionGroup</th>
      <th>CharlsonIndex</th>
      <th>ProcedureGroup</th>
      <th>SupLOS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Type</th>
      <td>int64</td>
      <td>float64</td>
      <td>float64</td>
      <td>float64</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>object</td>
      <td>int64</td>
    </tr>
    <tr>
      <th>0</th>
      <td>42286978</td>
      <td>8013252.0</td>
      <td>172193.0</td>
      <td>37796.0</td>
      <td>Y1</td>
      <td>Surgery</td>
      <td>Office</td>
      <td>28</td>
      <td>NaN</td>
      <td>8- 9 months</td>
      <td>NEUMENT</td>
      <td>0</td>
      <td>MED</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>97903248</td>
      <td>3316066.0</td>
      <td>726296.0</td>
      <td>5300.0</td>
      <td>Y3</td>
      <td>Internal</td>
      <td>Office</td>
      <td>50</td>
      <td>NaN</td>
      <td>7- 8 months</td>
      <td>NEUMENT</td>
      <td>1-2</td>
      <td>EM</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2759427</td>
      <td>2997752.0</td>
      <td>140343.0</td>
      <td>91972.0</td>
      <td>Y3</td>
      <td>Internal</td>
      <td>Office</td>
      <td>14</td>
      <td>NaN</td>
      <td>0- 1 month</td>
      <td>METAB3</td>
      <td>0</td>
      <td>EM</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73570559</td>
      <td>7053364.0</td>
      <td>240043.0</td>
      <td>70119.0</td>
      <td>Y3</td>
      <td>Laboratory</td>
      <td>Independent Lab</td>
      <td>24</td>
      <td>NaN</td>
      <td>5- 6 months</td>
      <td>METAB3</td>
      <td>1-2</td>
      <td>SCS</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11837054</td>
      <td>7557061.0</td>
      <td>496247.0</td>
      <td>68968.0</td>
      <td>Y2</td>
      <td>Surgery</td>
      <td>Outpatient Hospital</td>
      <td>27</td>
      <td>NaN</td>
      <td>4- 5 months</td>
      <td>FXDISLC</td>
      <td>1-2</td>
      <td>EM</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

Missing data:
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>LengthOfStay</th>
      <td>2597392</td>
      <td>0.973174</td>
    </tr>
    <tr>
      <th>DSFS</th>
      <td>52770</td>
      <td>0.019772</td>
    </tr>
    <tr>
      <th>Vendor</th>
      <td>24856</td>
      <td>0.009313</td>
    </tr>
    <tr>
      <th>ProviderID</th>
      <td>16264</td>
      <td>0.006094</td>
    </tr>
    <tr>
      <th>PrimaryConditionGroup</th>
      <td>11410</td>
      <td>0.004275</td>
    </tr>
    <tr>
      <th>Specialty</th>
      <td>8405</td>
      <td>0.003149</td>
    </tr>
    <tr>
      <th>PlaceSvc</th>
      <td>7632</td>
      <td>0.002860</td>
    </tr>
    <tr>
      <th>PCP</th>
      <td>7492</td>
      <td>0.002807</td>
    </tr>
    <tr>
      <th>ProcedureGroup</th>
      <td>3675</td>
      <td>0.001377</td>
    </tr>
    <tr>
      <th>SupLOS</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CharlsonIndex</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PayDelay</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>MemberID</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAZIAAAFGCAYAAABALRnDAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3de3wU9b3/8dcmZJOsEnIK0VKVKiCKhYNWmhA2CwlqIBFFCFYqhShCm5SWWor08LOnEglQ7PECisCh3qoHsWoRjWAUFEwoCQKnRY9YDhcFBQQvJJHcM9/fH3MSSNmNwO4sZnk/H499QGZ25z3fyW4+O/Od+Y7LGGMQERE5TVFnegVERKR9UyEREZGgqJCIiEhQVEhERCQoKiQiIhIUFRIREQmKCom0Kx9//DG9e/dmxIgRjBgxghtuuIExY8awatWqM7ZO8+fP56WXXnI855577mHIkCE8+OCDTJo0iZ07d57S6ydMmMAXX3xx0s9/+OGHuffee9t8zrvvvsuUKVNOaT0k8nQ40ysgcqri4uJYuXJly8+ffPIJt912G9HR0QwdOjTs6/PLX/4yLDnPPfcc69at49vf/vZpvX7Dhg0hXiPo27cvCxYsCPlypX3RHom0exdccAFTpkzhscceA2DPnj3cfvvt/PCHPyQjI4P8/Hzq6up4+eWXGTNmTMvr9u/fT1paGvX19SxYsIAbbriBUaNGcccdd3Do0KETcjZv3szo0aMZNWoUo0aNori4GIB/+7d/a8nu27cvDz/8MGPGjGHIkCEsW7as5fVLlixh2LBhDB8+nMmTJ1NVVQXA888/z6hRo7jpppu47bbb2LVr1wnZt956K8YYJk2axObNmxkyZAjvvvsu5eXl3HjjjYwZM4YbbriBqqoqpkyZwogRIxg5ciS//e1vsSyLGTNmAJCbm8uBAwdaLbuxsZG5c+cydOhQsrOzufvuu6mvr2/1nLfeeosxY8YwatQo0tPTeeihhwAoLy9n+PDhLdth5syZ3HrrrQwdOpR58+axZMkSxowZwzXXXMPGjRtP4bcq7YoRaUf27dtnrrzyyhOm79ixw/Tr188YY8zvf/9789JLLxljjKmvrzfDhw83r732mqmrqzOpqalmx44dxhhjHnroIfMf//EfZv/+/eb73/++qaurM8YY89hjj5k33njjhIzx48eboqIiY4wx27dvNzNnzjTGGPOb3/zG/PGPfzTGGNOrVy/z9NNPG2OMeffdd02fPn1MbW2tWbNmjcnMzDRHjhwxxhgzZ84c8+ijj5ry8nJz6623murqamOMMSUlJWbYsGF+296rVy/z+eefG2OMycjIMNu2bTNlZWXm8ssvNx9//LExxpgVK1aYCRMmGGOMaWxsNHfffbf58MMPT3j98Z566ikzduxYU1NTY5qamswvf/lLs2LFCrNgwQJTUFBgLMsyP/7xj82ePXuMMcYcPHjQ9O7d23z++eemrKzMXH/99S3b4eabbzb19fXm0KFDplevXuZPf/qTMcaYJ5980tx+++1+2yXtnw5tSURwuVzExcUBcNddd7FhwwaWLl3Khx9+yKFDh6iursbtdnPzzTfz/PPP85vf/IYVK1bw9NNPc/7553P55ZczcuRIBg0axKBBg0hNTT0hIysri3vvvZc333yTgQMHMnXqVL/rcs011wDwve99j/r6eqqrq9m4cSPDhg2jU6dOAC17CPfddx8fffRRqz2lyspKjhw5QmJi4km1vWvXrlxwwQUAXH311Tz44IOMGzeOgQMHkpuby3e/+902X//Xv/6VESNGtGy/5r2Nhx9+uGXbLl68mHXr1lFUVMSuXbswxlBTU3PCsjIyMoiJiSEpKQmPx4PP5wOgW7duHDly5KTaI+2PColEhHfffZdevXoBMHXqVJqamsjKyiI9PZ0DBw5g/m9IuTFjxjB69GiSk5O59NJLueiiiwB45plnePfdd9m4cSNz5szB5/Mxffr0VhljxowhIyODDRs2UFJSwiOPPMJrr712wrrExsYC9h9gAGMM0dHRLT+DXSwqKyuxLIsRI0Zw1113AWBZFocOHWopOCfD4/G0/P+iiy7ijTfeoLy8nLKyMm6//XbuvfdehgwZEvD1HTq0/jPw2WefYVlWy8/V1dWMHDmSa6+9lv79+5OTk8OaNWtatunx3G53m8uWyKQ+Emn39uzZw6OPPsqECRMAKC0tZfLkyWRnZwPw97//naamJsD+9n7llVcyZ84cfvSjHwHwwQcfMHz4cHr06MFPf/pTbrvtNt59990TcsaMGcP27dsZNWoUs2bNorKyksOHD5/UOg4cOJA33niDr776CrC/7T/55JOkpaXx6quvtvTJPPvss+Tm5p72tli2bBkzZswgLS2Nu+66i7S0NN5//30AoqOjaWxsPOE1qampFBUVUV9fj2VZzJw5k1dffbVl/kcffcRXX33FnXfeyZAhQygvL295rghoj0TaodraWkaMGAFAVFQUsbGxTJ06lfT0dAB+9atfMXnyZDweD+eeey4/+MEP2Lt3b8vrmwvB4MGDAbj88svJysoiJycHj8dDXFwcv/3tb0/InTZtGnPmzOGhhx7C5XLx85//nAsvvPCk1nnw4MHs3LmzpXj17NmTWbNmce655zJp0iQmTJiAy+Xi3HPP5ZFHHmm193IqbrrpJjZt2kR2djbx8fF07dqVcePGATBs2DDGjRvHww8/3LL3BnaB/OSTTxg1ahTGGJKTkxk3bhyLFi0C4LLLLiM9PZ2srCzcbje9evWiZ8+efPTRRyfsgcjZyWX87Z+KRCjLsrj33nv5zne+w09+8pMzvToiEUGHtuSs8dVXX5GSksKBAwcYP378mV4dkYihPRIREQmK9khERCQoKiQiIhKUs/KsLcuyaGo69SN60dGu03rd6VJe+82L5LYp7+zNi4mJ9jv9rCwkTU2GI0eqT/l1iYme03rd6VJe+82L5LYp7+zNS0rq6He6Dm2JiEhQVEhERCQoKiQiIhIUFRIREQmKComIiARFhURERIKiQiIiIkFRIRERkaA4UkiefPJJ0tPTSU9PZ8CAAcTFxbF582bS0tLw+Xzk5+e33BRn6dKl9O/fnwEDBlBUVARATU0NOTk5+Hw+srOzW24eVFZWRkpKCl6vl4KCgpa8goICkpOTGThwIJs2bXKiSSIiEoDjo/9OnjyZfv36UVRU1HLzoby8PIYOHUpqairXXXcdmzdvpra2lrS0NDZv3szChQuprKxk5syZLF++nI0bNzJ//nyuvPJKXnzxRbp37871119PYWEhYN9waO3atezbt4+cnBzeeeedNtepoaEp4FWdcVFfEdVU5XdebFwMdbUNfudZ0R2ptc49hS3z9drL1a7KO7NZylNeuPICXdnu6BApmzdv5n/+539YuHAhBQUFLXeky8rK4vXXXyc6Ohqv10tsbCyxsbH07NmTbdu2UVpa2nK/7KysrJbbmtbV1dGjRw8Ahg4dytq1a4mNjSUzMxOXy0W3bt1obGzk8OHDJCUlndY6RzVVYX1y4n24AfC4sarr/b/ugmHgCm0hERFpDxwtJHPmzOGee+4BwBjTcvvQjh07UlFRQWVlJZ06dWp5vr/px09LSEho9dzdu3cTFxdH586dT1hGW4UkOtpFYqLH7zxXTQx4/N8+NMrlIj7APOJicMf7X+bpio6OCrieTlBe+8xSnvLOdJ5jheTIkSN88MEHZGRkAPa9tZtVVVWRmJhIQkICVVVVbU7/uue63W6/09vS1qCNHtMQcK8j3uOmJtAeSW0D1XWh3TVtL7u7yjuzWcpTXrjywj5o49tvv821117b8vNVV13FunXrAFi9ejU+n4/k5GRKSkqora2loqKC7du306dPH7xeL6tWrWr13ISEBNxuN7t27cIYQ3FxMT6fD6/XS3FxMZZlsXfvXizLokuXLk41S0RE/oljeyT/+Mc/6N69e8vP999/P5MmTaK+vp7evXszevRooqOjmTJlCj6fD8uymD17NnFxceTn55Obm0taWhput5tly5YBsHjxYsaOHUtTUxOZmZmkpKQA4PP5SE1NxbIsFi5c6FSTRETEj7Pynu1tnbXlMQcCdra3eWjrgmFUu7qGbB2h/ezuKu/MZilPeeHK0/1IRETEESokIiISFBUSEREJigqJiIgERYVERESCokIiIiJBUSEREZGgqJCIiEhQVEhERCQoKiQiIhIUFRIREQmKComIiARFhURERIKiQiIiIkFRIRERkaCokIiISFBUSEREJCgqJCIiEhQVEhERCYoKiYiIBEWFREREgqJCIiIiQXGskMydO5fU1FSuvvpqHnvsMXbu3ElaWho+n4/8/HwsywJg6dKl9O/fnwEDBlBUVARATU0NOTk5+Hw+srOzOXz4MABlZWWkpKTg9XopKChoySooKCA5OZmBAweyadMmp5okIiJ+OFJI1q1bx1//+lc2bNjA+vXr2bdvH1OnTqWwsJCSkhKMMaxcuZKDBw+yYMECNmzYQHFxMTNmzKCuro5FixbRt29fSkpKGD9+PIWFhQDk5eWxbNkySktLKS8vZ+vWrWzdupX169dTXl7O8uXLmTx5shNNEhGRADo4sdDi4mL69u3LyJEjqays5A9/+ANLly5l8ODBAGRlZfH6668THR2N1+slNjaW2NhYevbsybZt2ygtLWX69Oktz501axaVlZXU1dXRo0cPAIYOHcratWuJjY0lMzMTl8tFt27daGxs5PDhwyQlJQVcv+hoF4mJHr/zXDUx4HH7nRflchEf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YmJiiImJcXSlRESk/TipPZKrr76aKVOmcPXVV7Nlyxauuuoqp9dLRETaia8tJM899xxTp05lw4YNvPfeeyQnJ/PjH/84HOsmIiLtQJuHth5++GE2bNhAY2Mj6enp3HTTTZSVlWmUXRERadFmIXn77beZP39+y73XL7zwQh588EHefPPNsKyciIh887VZSDweT8stcpvFxMRwzjnnOLpSIiLSfrRZSOLi4ti3b1+rafv27TuhuIiIyNmrzc72adOm8bOf/YzU1FQuuugi9u/fT2lpKfPmzQvX+omIyDdcm3skl156KcuWLeOKK66gpqaG733vezz77LNcccUV4Vo/ERH5hvva0387duzITTfdFI51ERGRduikrmwXEREJRIVERESC4kghaWpqYsKECXi9XgYNGsSuXbvYuXMnaWlp+Hw+8vPzsSwLgKVLl9K/f38GDBhAUVERADU1NeTk5ODz+cjOzubw4cMAlJWVkZKSgtfrpaCgoCWvoKCA5ORkBg4cyKZNm5xokoiIBHBSY22dqldeeQWADRs2sG7dOqZOnYoxhsLCQtLT08nLy2PlypWkpqayYMECNm/eTG1tLWlpaVx33XUsWrSIvn37MnPmTJYvX05hYSHz588nLy+PF198ke7du3P99dezdetWANavX095eTn79u0jJyeHd955x4lmiYiIH47skdx0003853/+JwAfffQR559/Plu2bGHw4MEAZGVlsWbNGjZt2oTX6yU2NpZOnTrRs2dPtm3bRmlpKcOGDWv13MrKSurq6ujRowcul4uhQ4eydu1aSktLyczMxOVy0a1bNxobG1v2YERExHmO7JEAdOjQgdzcXFasWMELL7xAUVFRy4WMHTt2pKKigsrKSjp16tTyGn/Tj5+WkJDQ6rm7d+8mLi6Ozp07n7CMpKSkgOsWHe0iMdHjd56rJgY8br/zolwu4gPMIy4Gd7z/ZZ6u6OiogOvpBOW1zyzlKe9M5zlWSACeeuop5s2bR0pKCjU1NS3Tq6qqSExMJCEhgaqqqjanf91z3W633+ltaWoyHDlS7XeexzRgVdf7nRfvcVMTYF5UbQPVdf6XeboSEz0B19MJymufWcpTXrjykpI6+p3uyKGtp59+mrlz5wL2eF1RUVH079+fdevWAbB69Wp8Ph/JycmUlJRQW1tLRUUF27dvp0+fPni9XlatWtXq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%0A"

# 2.Feature Engineering


# 3.Modelling


# 4.Evaluation
