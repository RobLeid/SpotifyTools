import pandas as pd
from io import BytesIO
import streamlit as st


def to_excel(df):
    """Convert DataFrame to Excel format for download"""
    if df is None:
        st.error("❌ Cannot create Excel file from None dataset")
        return None

    if df.empty:
        st.warning("⚠️ Dataset is empty - no Excel file generated")
        return None

    output = BytesIO()

    try:
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")

            # Add basic formatting
            workbook = writer.book
            worksheet = writer.sheets["Data"]

            # Header formatting
            header_format = workbook.add_format(
                {
                    "bold": True,
                    "text_wrap": True,
                    "valign": "top",
                    "fg_color": "#D7E4BC",
                    "border": 1,
                }
            )

            # Apply header formatting
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)

            # Define specific column widths for better initial viewing
            column_widths = {
                "Available Markets": 20,  # Standard width for markets
                "Unavailable Markets": 20,  # Standard width for markets
                "℗ Line": 20,  # Standard width for P Line
                "Album Spotify URL": 30,  # Standard width for URLs
                "Track Spotify URL": 30,  # Standard width for URLs
            }

            # Auto-adjust column widths with special handling for specific columns
            for i, col in enumerate(df.columns):
                if col in column_widths:
                    # Use predefined width for specific columns
                    worksheet.set_column(i, i, column_widths[col])
                else:
                    # Auto-adjust for other columns
                    max_length = max(df[col].astype(str).map(len).max(), len(col))
                    # Cap the width to prevent extremely wide columns
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.set_column(i, i, adjusted_width)

        output.seek(0)
        return output

    except Exception as e:
        st.error(f"❌ Error creating Excel file: {str(e)}")
        # Fallback to basic Excel without formatting
        return to_excel_basic(df)


def to_excel_basic(df):
    """Basic Excel generation without formatting (fallback)"""
    if df is None or df.empty:
        st.error("❌ Cannot create basic Excel file from empty/None dataset")
        return None

    output = BytesIO()

    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data")

        output.seek(0)
        return output

    except Exception as e:
        st.error(f"❌ Error creating basic Excel file: {str(e)}")
        return None
