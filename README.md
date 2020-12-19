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
</div>
</div>

</div>

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


Datatype:

<table class="dataframe" border="1">
  <thead>
    <tr>
      <th>Colum</th>
      <th>Type</th>>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MemberID</th>
      <td>int64</td>
    </tr>
    <tr>
      <th>ProviderID</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Vendor</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>PCP</th>
      <td>float64</td>
    </tr>
    <tr>
      <th>Year</th>
      <td>object</td>
    </tr>
    <tr>
      <th>Specialty</th>
      <td>object</td>
    </tr>
    <tr>
      <th>PrimaryConditionGroup</th>
      <td>object</td>
    </tr>
    <tr>
      <th>PlaceSvc</th>
      <td>object</td>
    </tr>
    <tr>
      <th>ProcedureGroup</th>
      <td>object</td>
    </tr>
    <tr>
      <th>CharlsonIndex</th>
      <td>object</td>
    </tr>
    <tr>
      <th>PayDelay</th>
      <td>object</td>
    </tr>
    <tr>
      <th>LengthOfStay</th>
      <td>object</td>
    </tr>
    <tr>
      <th>DSFS</th>
      <td>object</td>
    </tr>
    <tr>
      <th>SupLOS</th>
      <td>int64</td>
    </tr>  
  </tbody>
</table>


# 2.Feature Engineering


# 3.Modelling


# 4.Evaluation
