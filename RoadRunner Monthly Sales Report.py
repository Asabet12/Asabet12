#!/usr/bin/env python
# coding: utf-8

# #### <span style="color: blue; font-size: 18px;">**RoadRunner Tracker Closing of Printsmith**</span>
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />

# In[2]:


#Importing Libraries
import pandas as pd                   
import matplotlib.pyplot as plt       
import seaborn as sns                 
import numpy as np                    
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split
import sklearn.linear_model


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# specifying file name
file = r'C:\Users\Asabe\Downloads\Total Sales for May.xlsx'


# reading the file into Python
September_Closing = pd.read_excel(file)


# In[3]:



# Creating a function to find the desired text
def find_text_in_dataframe(df, target_text):
    found_text = []
    for column in df.columns:
        for value in df[column]:
            if isinstance(value, str) and any(phrase.lower() in value.lower() for phrase in target_text):
                found_text.append(value)
    return found_text

# Call the function to find occurrences of "total sales without postage" and "sales without postage" in the dataframe
target_text = ["total sales without postage", "Sales without Postage:"]
found_text = find_text_in_dataframe(September_Closing, target_text)

print("Occurrences of 'total sales without postage' and 'Sales without Postage':")
for text in found_text:
    print(text)

SeptemberSales_refined = pd.DataFrame({'Occurrences of "total sales without postage" and "Sales without Postage"': found_text})

    


# <span style="color: blue; font-size: 18px;">**Data Transformation and Massaging**</span>
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />

# In[4]:


SeptemberSales_refined.info(verbose =True)


# In[5]:


import re
import pandas as pd

def find_text_in_dataframe(df, target_text):
    found_text = []
    for column in df.columns:
        for value in df[column]:
            if isinstance(value, str) and target_text.lower() in value.lower():
                found_text.append(value)
    return found_text

# Call the function to find occurrences of "total sales without postage" in the dataframe
target_text = "total sales without postage"
found_text = find_text_in_dataframe(September_Closing, target_text)

# Initialize lists to store data for each column
job_estimate = []
job_description = []
customer_name = []
job_qty = []
total_sales_without_postage = []

# Regular expression pattern to find the total sales amount
total_sales_pattern = r"Total Sales without Postage: \$([\d,\.]+)"

# Regular expression pattern to find the customer name and job quantity
customer_name_job_qty_pattern = r"#([\s\S]+)\s+(\d[\d,]+)\s+Total Sales without Postage:"

# Process each found text and extract data into respective lists
for text in found_text:
    # Extract data using regular expressions
    job_estimate_match = re.search(r"\b\d+\b", text)
    job_estimate.append(job_estimate_match.group() if job_estimate_match else None)

    job_description_text = text[job_estimate_match.end():].strip()
    # Extract only the first part of the job description before "#"
    job_description_match = re.search(r"(.+?)(?=#\d)", job_description_text)
    job_description.append(job_description_match.group(1).strip() if job_description_match else None)

    customer_name_job_qty_match = re.search(customer_name_job_qty_pattern, text)
    if customer_name_job_qty_match:
        customer_name_text = customer_name_job_qty_match.group(1).strip()
        # Remove numeric digits from the customer name
        customer_name_text = re.sub(r"\d+", "", customer_name_text).strip()
        # Extract only the numeric part from the job quantity
        job_qty_match = re.search(r"\b\d+\b", customer_name_job_qty_match.group(2))
        job_qty_text = job_qty_match.group() if job_qty_match else None
    else:
        # If customer name is not found, extract everything after "#"
        customer_name_match = re.search(r"#([\s\S]+)", text)
        customer_name_text = customer_name_match.group(1).strip() if customer_name_match else None
        if customer_name_text is not None:
            # Remove numeric digits from the customer name
            customer_name_text = re.sub(r"\d+", "", customer_name_text).strip()
        # Extract only the numeric part from the job quantity
        job_qty_match = re.search(r"\b\d+\b", job_description_text)
        job_qty_text = job_qty_match.group() if job_qty_match else None

    customer_name.append(customer_name_text)
    job_qty.append(job_qty_text)

    total_sales_match = re.search(total_sales_pattern, text)
    total_sales_without_postage.append(total_sales_match.group(1) if total_sales_match else None)

# Create the July_Sales_refined dataframe
September_Sales_refined = pd.DataFrame({
    'Job Estimate': job_estimate,
    'Job Description': job_description,
    'Customer Name': customer_name,
    'Job Qty': job_qty,
    'Total Sales without Postage': total_sales_without_postage
})

# Remove "Total Sales without Postage: $" from the "Customer Name" column
September_Sales_refined["Customer Name"] = September_Sales_refined["Customer Name"].str.replace(r"Total Sales without Postage: \$", "")

# Print the new dataframe
print(September_Sales_refined.head(n=10))


# <span style="color: blue; font-size: 18px;">**Adding Joseph's Accounts**</span>
# <hr style="height:.9px;border:none;color:#333;background-color:#333;" />

# In[6]:


# List of Joseph's clients
Joseph_clients = [
    "AVATIER CORPORATION",
    "CALIFORNIA SYMPHONY",
    "CALIFORNIA SYMPHONY ALLIANCE",
    "DONOR NETWORK WEST",
    "LIVERMORE VALLEY JOINT UNIFIED SCHOOL DISTRICT",
    "MARRIOTT HOTELS",
    "RMS Manufacturing, BCMP North, LLC",
    "SAN DAMIANO RETREAT",
    "ST. AGUSTINE'S EPISCOPAL CHURCH",
    "TAYLOR MORRISON",
    "TRUMARK",
    "Superior Roof Metals",
    "HEWITT, JONES & FITCH CPA",
    "RUTHERFORD + CHEKENE ENGINEERS",
    "Oakmore Homes Association",
    "eCIFM",
    "Be Clinical",
    "East Bay Regional Park District"
]

# Function to determine client source
def determine_client_source(customer_name):
    if customer_name in Joseph_clients:
        return "Joseph's Client"
    else:
        return "Roadrunner's Client"

# Add a new column "Client Source" to the July_Sales_refined DataFrame
September_Sales_refined["Client Source"] = September_Sales_refined["Customer Name"].apply(determine_client_source)


# In[7]:


September_Sales_refined.info(verbose = True)


# In[8]:


import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Function to send an email with an attachment
def send_email_with_attachment(filename, recipient_email):
    # Email configurations
    sender_email = "asabett12@gmail.com"
    sender_password = "sM88KzYr2$"
    smtp_server = "smtp.gmail.com"  # Replace with your SMTP server address
    smtp_port = 587  # Replace with your SMTP server port

    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = "July Sales Report"

    # Attach the Excel file to the email
    with open(filename, "rb") as file:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(filename)}")
        message.attach(part)

    # Connect to the SMTP server and send the email
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

# File path of the Excel file
file_name = 'October_Sales_refined.xlsx'
current_working_directory = os.getcwd()
file_path = os.path.join(current_working_directory, file_name)
http://localhost:8888/notebooks/RoadRunner%20Monthly%20Sales%20Report.ipynb#
# Assuming "July_Sales_refined" is the DataFrame containing the data
# You may need to adjust this part based on how you obtained the "July_Sales_refined" DataFrame
# For example, if "July_Sales_refined" is a global variable, you can directly use it here.
# Otherwise, you may need to load the Excel file and create the DataFrame before this step.

# Export the DataFrame to Excel
October_Sales_refined.to_excel(file_path, index=False)

# Recipient's email address
recipient_email = "gm@roadrunnerprintmail.com"

# Send the email with the attachment
send_email_with_attachment(file_path, recipient_email)


# In[9]:


import os
file_name = 'September_Sales_refined.xlsx'
current_working_directory = os.getcwd()
file_path = os.path.join(current_working_directory, file_name)
September_Sales_refined.to_excel(file_path, index=False)


# In[ ]:




