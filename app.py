import streamlit as st
import pandas as pd

st.title("CSV Email Matcher")

# File Uploaders
specific_emails = st.file_uploader("Upload Specific Emails CSV", type="csv")
full_db = st.file_uploader("Upload Full Database CSV", type="csv")

if specific_emails and full_db:
    if st.button("Process"):
        try:
            # Read files
            specific_df = pd.read_csv(specific_emails)
            full_df = pd.read_csv(full_db)

            # Clean columns
            specific_df.columns = specific_df.columns.str.strip()
            full_df.columns = full_df.columns.str.strip()

            # Validate email columns
            if 'email' not in specific_df.columns or 'email' not in full_df.columns:
                st.error("Both files must have an 'email' column!")
            else:
                # Match data
                matched_data = full_df[full_df['email'].isin(specific_df['email'])]

                # Download button
                st.success(f"Found {len(matched_data)} matching records!")
                st.download_button(
                    label="Download Matched Data",
                    data=matched_data.to_csv(index=False).encode('utf-8'),
                    file_name="matched_data.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error: {str(e)}")