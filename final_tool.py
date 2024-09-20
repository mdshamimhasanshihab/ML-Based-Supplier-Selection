import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import seaborn as sns

# Load data and preprocess if necessary
filtered_data = pd.read_excel("Filtered_data.xlsx")
performance_df = pd.read_excel("F_Total.xlsx")
risk_data = pd.read_excel("vendor_risk.xlsx")
summary_stats = filtered_data.describe().T
cmap = sns.diverging_palette(2, 165, s=80, l=55, n=9)
m_filtered_data = filtered_data.drop(columns=['Item Name', 'EWO No', 'Order Date', 'Delivery Date', 'Supplier Address'])
X = m_filtered_data.drop(columns=['Vendor'])
y = m_filtered_data['Vendor']
y = y.str.replace('V', '')
y = y.astype(int)
col_names = list(X.columns)

# Standardize the input data
s_scaler = preprocessing.StandardScaler()
X_scaled = s_scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=col_names)

# Train a RandomForestClassifier model
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model2.fit(X_scaled, y)

# Streamlit app interface
st.title('Supplier Evaluation Form')

# Expander for selecting number of supplier suggestions
with st.expander("Number of Supplier Suggestions"):
    sg = int(st.selectbox("How many suppliers do you want to suggest?", [1, 2, 3, 4, 5], index=2))

# Expander for inputting values to the ML model
with st.expander("Input Values for Supplier Evaluation"):
    res = int(st.selectbox("Responsiveness [1~5]", [1, 2, 3, 4, 5], index=2))
    coop = int(st.selectbox("Cooperativeness [1~5]", [1, 2, 3, 4, 5], index=2))
    pack = int(st.selectbox("Packaging [1~5]", [1, 2, 3, 4, 5], index=2))
    con = int(st.selectbox("Consistency [1~5]", [1, 2, 3, 4, 5], index=2))
    dlp = int(st.selectbox("Delivery Performance [1~5]", [1, 2, 3, 4, 5], index=2))
    fin = int(st.selectbox("Fineness [1~5]", [1, 2, 3, 4, 5], index=2))
    flex = int(st.selectbox("Flexibility [1~5]", [1, 2, 3, 4, 5], index=2))
    afs = int(st.selectbox("After Sale Service [1~5]", [1, 2, 3, 4, 5], index=2))
    odf = int(st.selectbox("Order Fulfillment [0/1]", [0, 1], index=1))
    ont = int(st.selectbox("On Time Delivery [0/1]", [0, 1], index=1))
    leng = float(st.number_input("Yarn Length (cm)", value=110.0))
    strn = float(st.number_input("Yarn Strength (gm/denier)", value=4))
    qnt = int(st.number_input("Quantity (Kg)", value=60))
    unt = float(st.number_input("Unit Price", value=2))
    ltd = int(st.number_input("Lead Time (days)", value=5))

# Function to standardize and transform inputs
def transform_input(value, mean, std):
    return (value - mean) / std

# Standardize inputs based on summary statistics
res = transform_input(res, summary_stats.loc["Responsiveness", "mean"], summary_stats.loc["Responsiveness", "std"])
coop = transform_input(coop, summary_stats.loc["Co_operativeness", "mean"], summary_stats.loc["Co_operativeness", "std"])
pack = transform_input(pack, summary_stats.loc["Packaging", "mean"], summary_stats.loc["Packaging", "std"])
con = transform_input(con, summary_stats.loc["Consistency", "mean"], summary_stats.loc["Consistency", "std"])
ont = transform_input(ont, summary_stats.loc["On time Delivery", "mean"], summary_stats.loc["On time Delivery", "std"])
dlp = transform_input(dlp, summary_stats.loc["Delivery Performance", "mean"], summary_stats.loc["Delivery Performance", "std"])
fin = transform_input(fin, summary_stats.loc["Fineness", "mean"], summary_stats.loc["Fineness", "std"])
flex = transform_input(flex, summary_stats.loc["Flexibility", "mean"], summary_stats.loc["Flexibility", "std"])
odf = transform_input(odf, summary_stats.loc["Order Fullfillment", "mean"], summary_stats.loc["Order Fullfillment", "std"])
afs = transform_input(afs, summary_stats.loc["After sale servixe", "mean"], summary_stats.loc["After sale servixe", "std"])

# Execute when 'Suggest Suppliers' button is clicked
if st.button('Suggest Suppliers 🛠️', key='suggest_suppliers'):
    results = []
    unique_results = set()
    for a in range(1000):
        if a % 2 == 0:
            leng1 = leng
            leng1 -= a * (51.630814 / 100)
            leng2 = transform_input(leng1, summary_stats.loc["Length(cm)", "mean"], summary_stats.loc["Length(cm)", "std"])

            strn1 = strn
            strn1 -= a * (0.312247 / 100)
            strn2 = transform_input(strn1, summary_stats.loc["Strength (gm/denier)", "mean"], summary_stats.loc["Strength (gm/denier)", "std"])

            qnt1 = qnt
            qnt1 -= a * (134.504297 / 100)
            qnt2 = transform_input(qnt1, summary_stats.loc["Total Qty in (Kg)", "mean"], summary_stats.loc["Total Qty in (Kg)", "std"])

            unt1 = unt
            unt1 -= a * (0.872881 / 100)
            unt2 = transform_input(unt1, summary_stats.loc["Unit price", "mean"], summary_stats.loc["Unit price", "std"])

            ltd1 = ltd
            ltd1 -= a * (4.3116941 / 100)
            ltd2 = transform_input(ltd1, summary_stats.loc["Lead Time", "mean"], summary_stats.loc["Lead Time", "std"])
        else:
            leng1 = leng
            leng1 += a * (51.630814 / 100)
            leng2 = transform_input(leng1, summary_stats.loc["Length(cm)", "mean"], summary_stats.loc["Length(cm)", "std"])

            strn1 = strn
            strn1 += a * (0.312247 / 100)
            strn2 = transform_input(strn1, summary_stats.loc["Strength (gm/denier)", "mean"], summary_stats.loc["Strength (gm/denier)", "std"])

            qnt1 = qnt
            qnt1 += a * (134.504297 / 100)
            qnt2 = transform_input(qnt1, summary_stats.loc["Total Qty in (Kg)", "mean"], summary_stats.loc["Total Qty in (Kg)", "std"])

            unt1 = unt
            unt1 += a * (0.872881 / 100)
            unt2 = transform_input(unt1, summary_stats.loc["Unit price", "mean"], summary_stats.loc["Unit price", "std"])

            ltd1 = ltd
            ltd1 += a * (4.3116941 / 100)
            ltd2 = transform_input(ltd1, summary_stats.loc["Lead Time", "mean"], summary_stats.loc["Lead Time", "std"])

        tp = unt1 * qnt1
        tp = transform_input(tp, summary_stats.loc["Total Price", "mean"], summary_stats.loc["Total Price", "std"])

        # Predict using the model
        result = model2.predict([[res, coop, pack, con, ont, dlp, fin, leng2, strn2, flex, odf, tp, afs, qnt2, ltd2, unt2]])
        result_tuple = tuple(result)
        
        if result_tuple not in unique_results:
            unique_results.add(result_tuple)
            results.append(result)

        # Stop if we have reached the desired number of suggestions
        if len(unique_results) >= sg:
            break

    suggested_suppliers = []
    for vendor in unique_results:
        vendor_id = 'V' + str(vendor[0])
        rating = performance_df[performance_df['Vendor'] == vendor_id]['Total'].values[0]
        risk = risk_data[risk_data['Vendor'] == vendor_id]['overall_risk'].values[0]
        suggested_suppliers.append({'Vendor': vendor_id, 'Performance Rating': rating * 20, 'Risk Factor': risk * 100})

    suggested_suppliers_df = pd.DataFrame(suggested_suppliers)

    # Reset the index to start from 1
    suggested_suppliers_df.index = range(1, len(suggested_suppliers_df) + 1)

    # Display suggested suppliers
    st.subheader("Suggested Suppliers by ML and their Performance Ratings and Risk Factors:")
    st.table(suggested_suppliers_df)
# Merge performance and risk data
merged_df = pd.merge(performance_df, risk_data, on='Vendor')

# Multiply the performance ratings by 20 and the risk ratings by 100
merged_df['Total'] = merged_df['Total'] * 20
merged_df['overall_risk'] = merged_df['overall_risk'] * 100

# Expander for displaying top performers
with st.expander("Show Top 3 Highest Performers"):
    # Sort the merged data by performance rating and get the top 3
    top_performers = merged_df.sort_values(by='Total', ascending=False).head(3)
    
    # Rename columns
    top_performers = top_performers.rename(columns={'Total': 'Performance Rating', 'overall_risk': 'Risk Factor'})
    
    # Display top performers
    st.subheader("Top 3 Highest Performers:")
    st.table(top_performers[['Vendor', 'Performance Rating', 'Risk Factor']])

# Expander for displaying less risky vendors
with st.expander("Show Lowest 3 Risk Factors"):
    # Sort the merged data by overall risk and get the lowest 3
    lowest_risk = merged_df.sort_values(by='overall_risk').head(3)
    
    # Rename columns
    lowest_risk = lowest_risk.rename(columns={'Total': 'Performance Rating', 'overall_risk': 'Risk Factor'})
    
    # Display less risky vendors
    st.subheader("Less Risky Vendors:")
    st.table(lowest_risk[['Vendor', 'Performance Rating', 'Risk Factor']])
