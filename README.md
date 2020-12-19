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
    │   ├── New Data (store all data after processing in step 1,2,3,4,5)
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
<a href="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/tree/main/1.Exploration">Click here go to folder</a>

This is step exploring raw data and feature data after processing in step 2.
## Expore raw data
This step tells us the data structure, the data form, visualizate data so we know what to do with this data before we put it into model training.

Data dictionaty:

    Members Data Table
        MemberID: Member pseudonym.
        AgeAtFirstClaim: Age in years at the time of the first claim’s date of service computed froM the date of birth; Generalized into ten year age intervals.
        Sex: Biological sex of member: M = Male; F=Female.
    Claims (Level 2) Data
        MemberID: Member pseudonym.
        ProviderID: Provider pseudonym.
        Vendor: Vendor pseudonym.
        PCP: Primary care physician pseudonym.
        Year: Year in which the claim was made: Y1; Y2; Y3.
        Specialty: Generalized specialty.
        PlaceSvc: Generalized place of service.
        PayDelay: Number of days delay between the date of service (the date the actual procedure was performed or service provided) and date of payment 
        LengthOfStay: Length of stay (discharge date – admission date + 1)
        DSFS: Days since first claim, computed from the first claim for that member for each year.
        PrimaryConditionGroup: Broad diagnostic categories, based on the relative similarity of diseases and mortality rates
        CharlsonIndex: A measure of the affect diseases have on overall illness, grouped by significance, that generalizes additional diagnoses.
        ProcedureGroup: Broad categories of procedures, grouped according to the hierarchical structure defined by the Current Procedural Terminology (CPT) [3].
        SupLOS: Indicates if the NULL value for the LengthOfStay variable is due to suppression done during the de-identification process. A value of 1 indicates
        that suppression was applied.
    Drug Count Data
        Year: Year in which the drug prescription was filled: Y1; Y2; Y3.
        DSFS: Days since first service (or claim), computed from the first claim for that member for each year
        DrugCount: Count of unique prescription drugs filled by DSFS. 
    Lab Count Data
        Year: Year in which the drug prescription was filled: Y1; Y2; Y3.
        DSFS: Days since first service (or claim), computed from the first claim for that member for each year

    Outcome Data
        MemberID Member pseudonym.
        DaysInHospital_Y2: Days in hospital, the main outcome, for members with claims in Y1. Values above 14 days (the 99% percentile) are top-coded as “15+”.
        DaysInHospital_Y3 Days in hospital, the main outcome, for members with claims in Y2. Values above 14 days (the 99% percentile) are top-coded as “15+”.
        ClaimedTruncated Members with truncated claims in the year prior to the main outcome are assigned a value of 1, and 0 otherwise.


Giải thích thuật ngữ:

    Bảng Members Table
        MemberID: id của bệnh nhân (mỗi bệnh nhân chỉ có 1 id duy nhất cho tất cả các lần claim bảo hiểm)
        AgeAtFirstClaim: tuổi của bệnh nhân ghi nhận ở lần claim đầu tiên
        Sex: giới tính 
    Claims Table
        MemberID: id của bệnh nhân
        ProviderID: id của Bác sĩ / Nhân viên y tế chăm sóc cho bệnh nhân đó
        Vendor: đơn vị xuất hoá đơn viện phí (chẳng hạn bệnh viên, phòng khám,....)
        PCP: id của bác sĩ chăm sóc ban đầu của bệnh nhân
        Year: năm claim bảo hiểm (Y1, Y2, hay Y3)
        Specialty: chuyên khoa điều trị 
        PlaceSvc: nơi điều trị 
        PayDelay: thời gian từ lúc claim đến lúc được trả tiền
        LengthOfStay: thời gian nằm viện.
        DSFS (days since first service that year): số ngày kể từ lần sử dụng dịch vụ đầu tiên trong năm đó
        PrimaryConditionGroup (a generalization of the primary diagnosis codes): mã số nhóm bệnh được chuẩn đoán
        CharlsonIndex (a generalization of the diagnosis codes in the form of a categorized comorbidity score): một loại chỉ số đánh giá nguy cơ tử vong trong 10 năm
        ProcedureGroup (a generalization of the CPT code or treatment code): mã số nhóm điều trị
        SupLOS (suppressed Length of stay): đánh dấu LengthOfStay có null hay không
    Labs Table: chứa kết quả các chỉ số xét nghiệm.
    Drug Table: chứa thông tin các loại thuốc được kê đơn.
    DaysInHospital Tables: chứa thông tin về số ngày nằm viện của bệnh nhan.
        MemberID: id của bệnh nhân
        ClaimsTruncated: đánh dấu nếu claim của bệnh nhân bị truncated (giả sử hoá đơn của bệnh nhân là $1000, nhưng bảo hiểm chỉ chi trả $800 thì claim đó là truncated). Nếu biến này là 1 cho một bệnh nhân nào đó trong DaysInHospital_Y2 thì có nghĩa là bệnh nhân đó đã từng có truncated trong Y1.
        DaysInHospital: số ngày nằm viện trong từng năm.

### Input data: Claims.csv 
shape(2668990, 14)

#### Days since first claim:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/DSFS.png">

#### PrimaryConditionGroup:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/PrimaryConditionGroup.png">

#### Generalized specialty:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/Specialty.png">

#### Broad diagnostic categories:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/ProcedureGroup.png">

### Input data: Members.csv
#### AgeAtFirstClaim:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/AgeAtFirstClaim.png">

#### Sex:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/Sex_member.png">

### Input data: LabCount.csv
#### LabCount:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/LabCount.png">

### Input data: DrugCount.csv
#### LabCount:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/DrugCount.png">

### Outcome: DaysInHospital_Y2.csv
#### DaysInHospital:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/DIH.png">

#### ClaimsTruncated:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/ClaimTruncated.png">

### Outcome: DaysInHospital_Y3.csv
#### DaysInHospital:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/DIH_Y3.png">

#### ClaimsTruncated:
<img src="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/blob/main/imgsrc/ClaimTruncated_Y3.png">

## Explore Feature after process
This step tells us the data structure, the data form, visualizate data after finished step 2.

<table class="dataframe" border="1">
  <thead>
    <tr>
      <th></th>
      <th>Year</th>
      <th>MemberID</th>
      <th>LabCount_total</th>
      <th>LabCount_max</th>
      <th>LabCount_min</th>
      <th>LabCount_ave</th>
      <th>LabCount_months</th>
      <th>LabCount_std</th>
      <th>DrugCount_total</th>
      <th>DrugCount_max</th>
      <th>...</th>
      <th>ProcedureGroup_Count_SO</th>
      <th>ProcedureGroup_Count_SMCD</th>
      <th>AgeAtFirstClaim</th>
      <th>Male</th>
      <th>Female</th>
      <th>MissSex</th>
      <th>MissAge</th>
      <th>ClaimsTruncated</th>
      <th>TARGET</th>
      <th>trainset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Y1</td>
      <td>210</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Y2</td>
      <td>210</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Y3</td>
      <td>210</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>5.0</td>
      <td>2.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Y1</td>
      <td>3889</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>30.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Y1</td>
      <td>11951</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>




# 2.Feature Engineering 
<a href="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/tree/main/2.Feature%20Engineering">Click here go to folder</a>

## Members processing
AgeApprox: based on the following lookup table derived from AgeAtFirstClaim
<table class="dataframe" border="1">
  <thead>
    <tr>
      <th>AgeAtFirstClaim</th>
      <th>AgeApprox</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>#N/A</th>
      <td>45</td>
    </tr>
    <tr>
      <th>0-9</th>
      <td>5</td>
    </tr>
    <tr>
      <th>10-19</th>
      <td>15</td>
    </tr>
    <tr>
      <th>20-29</th>
      <td>25</td>
    </tr>
    <tr>
      <th>30-39</th>
      <td>35</td>
    </tr>
    <tr>
      <th>40-49</th>
      <td>45</td>
    </tr>
    <tr>
      <th>50-59</th>
      <td>55</td>
    </tr>
    <tr>
      <th>60-69</th>
      <td>68</td>
    </tr>
    <tr>
      <th>70-79</th>
      <td>75</td>
    </tr>
    <tr>
      <th>80+</th>
      <td>85</td>
    </tr>
    <tr>
      <th></th>
      <td></td>
    </tr>
    <tr>
      <th>Male</th>
      <td>=IF(SEX = "M",1,0)</td>
    </tr>
    <tr>
      <th>Female</th>
      <td>=IF(SEX = "F",1,0)</td>
    </tr>
    <tr>
      <th>MissSex</th>
      <td>=IF(SEX = "",1,0)</td>
    </tr>
  </tbody>
</table>

# 3.Modelling 
<a href="https://github.com/quangson05dt1/vef_FinalProject_Heritage-Health-Prize/tree/main/3.Modelling">Click here go to folder</a>


# 4.Evaluation 
<a href="https://drive.google.com/drive/folders/1pwQ3H4aJ8a6yyJHZkTwtjcL4wYWQb7bn">Click here go to folder</a>
