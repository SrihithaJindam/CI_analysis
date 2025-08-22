import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
import os

def round_to_5min(dt):
    """Round datetime to nearest 5 minutes"""
    remainder = dt.minute % 5 + dt.second / 60.0
    if remainder < 2.5:
        # Round down
        return dt - timedelta(minutes=remainder)
    else:
        # Round up
        return dt + timedelta(minutes=(5 - remainder))
    
    # Create all KPIs
    figures = {
        'KPI 1.1 - Matched Capacity vs Time': create_kpi_1_1(df),
        'KPI 1.2 - Relative Matched Capacity vs Time': create_kpi_1_2(df),
        'KPI 1.3 - Relative Matched Capacity per Allocation Time': create_kpi_1_3(df),
        'KPI 1.4 - Matched Capacity per Allocation Time': create_kpi_1_4(df),
        'KPI 2 - Available Capacity until Delivery': create_kpi_2(df),
        'KPI 3.1 - Capacity Utilization': create_kpi_3_1(df)
    }
    
    # Add KPI 3.2 figures (which return a list of dictionaries)
    for kpi_name, kpi_func, has_multiple_plots in [
        ('KPI 3.2', create_kpi_3_2, True)
    ]:
        if has_multiple_plots:
            results = kpi_func(df)
            for result in results:
                border = result['border']
                figures[f'{kpi_name} - Capacity Utilization Analysis - {border}'] = result['main_fig']
                figures[f'{kpi_name} - Utilization Statistics - {border}'] = result['stats_fig']
    
    # Save all plots to a single file
    save_all_plots(figures, save_path)
    
    return df

def load_data(file_path):
    """Load and preprocess the data with optimizations for large files and data cleaning"""
    try:
        print("\nStarting data loading and cleaning process...")
        
        # Read CSV with optimized settings for large files
        print("Reading CSV file...")
        df = pd.read_csv(
            file_path,
            parse_dates=['Delivery Start (CET)', 'Delivery End (CET)', 
                        'Allocation Time (CET)', 'Request Time (CET)'],
            date_parser=lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M:%S', dayfirst=True),
            dtype={
                'Matched Capacity': 'float32',
                'ATC with RR after Allocation': 'float32',
                'Allocation Type': 'category',
                'Direction': 'category'
            }
        )
        
        # Data Cleaning Steps
        print("\nCleaning data...")
        
        # 1. Remove completely empty rows
        initial_rows = len(df)
        df.dropna(how='all', inplace=True)
        empty_rows_removed = initial_rows - len(df)
        print(f"Removed {empty_rows_removed} completely empty rows")
        
        # 2. Clean string columns - remove extra spaces and standardize text
        string_columns = ['Direction', 'Allocation Type']
        for col in string_columns:
            if col in df.columns:
                # Remove leading/trailing spaces and convert to consistent case
                df[col] = df[col].astype(str).str.strip().str.upper()
                # Remove any multiple spaces between words
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                # Standardize direction format
                if col == 'Direction':
                    df[col] = df[col].str.replace('->|→|TO', '->', regex=True)
        
        # 3. Handle numeric columns
        numeric_columns = ['Matched Capacity', 'ATC with RR after Allocation']
        for col in numeric_columns:
            if col in df.columns:
                # Convert any non-numeric values to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Report number of invalid values
                invalid_count = df[col].isna().sum()
                if invalid_count > 0:
                    print(f"Found {invalid_count} invalid values in {col}")
                # Fill NaN with 0 for these columns as they represent capacities
                df[col].fillna(0, inplace=True)
                # Convert to float32 for memory efficiency
                df[col] = df[col].astype('float32')
        
        # 4. Clean and validate datetime columns
        print("\nValidating datetime columns...")
        time_columns = ['Delivery Start (CET)', 'Delivery End (CET)', 
                       'Allocation Time (CET)', 'Request Time (CET)']
        
        for col in time_columns:
            if col in df.columns:
                # Identify invalid datetime entries
                invalid_dates = df[col].isna()
                invalid_count = invalid_dates.sum()
                if invalid_count > 0:
                    print(f"Found {invalid_count} invalid dates in {col}")
                    # For datetime columns, drop rows with invalid dates as they are crucial
                    df = df[~invalid_dates]
                    print(f"Dropped {invalid_count} rows with invalid dates in {col}")
                
                # Round times to 5-minute intervals
                df[col] = df[col].apply(round_to_5min)
                
                # Ensure all dates are within a reasonable range
                min_date = df[col].min()
                max_date = df[col].max()
                print(f"{col} range: {min_date} to {max_date}")
        
        # 5. Validate data consistency
        print("\nValidating data consistency...")
        # Ensure Delivery End is after Delivery Start
        invalid_delivery = df['Delivery End (CET)'] <= df['Delivery Start (CET)']
        invalid_delivery_count = invalid_delivery.sum()
        if invalid_delivery_count > 0:
            print(f"Found {invalid_delivery_count} rows where Delivery End <= Delivery Start")
            df = df[~invalid_delivery]
            print(f"Dropped {invalid_delivery_count} rows with invalid delivery times")
        
        # 6. Calculate and validate time-related fields
        print("\nCalculating time-related fields...")
        # Calculate time to delivery in minutes
        df['Time to Delivery'] = (df['Delivery Start (CET)'] - df['Allocation Time (CET)']).dt.total_seconds() / 60
        df['Time to Delivery'] = df['Time to Delivery'].astype('float32')
        
        # Validate Time to Delivery
        invalid_delivery_time = df['Time to Delivery'] < 0
        invalid_delivery_time_count = invalid_delivery_time.sum()
        if invalid_delivery_time_count > 0:
            print(f"Found {invalid_delivery_time_count} rows with negative Time to Delivery")
            df = df[~invalid_delivery_time]
            print(f"Dropped {invalid_delivery_time_count} rows with invalid delivery times")
        
        # Round time to delivery to 5 minute intervals
        df['Time Bucket'] = (df['Time to Delivery'] // 5) * 5
        
        # 7. Filter for implicit allocations only
        initial_rows = len(df)
        print("\nUnique Allocation Types before filtering:", df['Allocation Type'].unique())
        df = df[df['Allocation Type'].str.upper().str.contains('IMPLICIT')]
        filtered_rows = initial_rows - len(df)
        print(f"Removed {filtered_rows:,} non-implicit allocation rows")
        print("Remaining Allocation Types:", df['Allocation Type'].unique())
        print("Remaining Directions:", df['Direction'].unique())
        
        # 8. Final validation
        print("\nFinal data validation:")
        print(f"Final number of rows: {len(df):,}")
        print(f"Available directions: {', '.join(df['Direction'].unique())}")
        print(f"Time period: {df['Delivery Start (CET)'].min()} to {df['Delivery Start (CET)'].max()}")
        print("\nValue ranges:")
        print(f"Matched Capacity: {df['Matched Capacity'].min():.2f} to {df['Matched Capacity'].max():.2f}")
        print(f"ATC with RR: {df['ATC with RR after Allocation'].min():.2f} to {df['ATC with RR after Allocation'].max():.2f}")
        print(f"Time to Delivery: {df['Time to Delivery'].min():.2f} to {df['Time to Delivery'].max():.2f} minutes")
        
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    
    return df

def create_kpi_1_1(df):
    """KPI 1.1 - Matched Capacity for implicit allocation per time to delivery start (absolute)"""
    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Process data for both directions
    for border in df_plot['Direction'].unique():
        border_data = df_plot[df_plot['Direction'] == border].copy()
        
        # For DK2->50HzT direction, make values negative
        if border.upper() == 'DK2->50HZT':
            border_data.loc[:, 'Matched Capacity'] = -border_data['Matched Capacity']
            print(f"Applying negative values for direction: {border}")
            print(f"Value range after negation: {border_data['Matched Capacity'].min():.2f} to {border_data['Matched Capacity'].max():.2f}")
        
        # Group by allocation time (rounded to 5 minutes)
        avg_capacity = border_data.groupby('Allocation Time (CET)')['Matched Capacity'].mean()
        
        # Add trace with custom styling
        is_dk2_to_50hzt = 'DK2->50HZT' in border.upper()
        line_color = 'rgb(31, 119, 180)' if is_dk2_to_50hzt else 'rgb(255, 127, 14)'
        trace_name = f"{border} ({'↓' if is_dk2_to_50hzt else '↑'})"
        
        fig.add_trace(go.Scatter(
            x=avg_capacity.index,
            y=avg_capacity.values,
            name=trace_name,
            mode='lines',
            line=dict(
                color=line_color,
                width=2
            ),
            hovertemplate='Time: %{x}<br>Direction: %{text}<br>Capacity: %{y:.2f} MW<extra></extra>',
            text=[border] * len(avg_capacity)
        ))
    
    # Print debug information
    print("\nDebug information for KPI 1.1:")
    print(f"Available directions: {df_plot['Direction'].unique()}")
    print("Value ranges per direction:")
    for direction in df_plot['Direction'].unique():
        dir_data = df_plot[df_plot['Direction'] == direction]['Matched Capacity']
        print(f"{direction}: {dir_data.min():.2f} to {dir_data.max():.2f}")
    
    # Calculate maximum absolute value for symmetric y-axis
    max_abs_val = max(
        abs(df_plot['Matched Capacity'].max()),
        abs(df_plot['Matched Capacity'].min())
    ) * 1.1  # Add 10% padding
    
    print(f"Y-axis range will be set to: -{max_abs_val:.2f} to {max_abs_val:.2f}")
    
    # Update layout with improved styling
    fig.update_layout(
        title=dict(
            text='KPI 1.1 - Matched Capacity vs Time (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Matched Capacity (MW)<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    return fig

def create_kpi_1_2(df):
    """KPI 1.2 - Matched capacity per time to delivery start (relative)"""
    fig = go.Figure()
    
    max_rel_val = 0  # Track maximum relative value for y-axis scaling
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border].copy()
        
        # Calculate relative capacity for each allocation time
        total_by_time = border_data.groupby('Allocation Time (CET)')['Matched Capacity'].sum()
        total_matched = abs(border_data['Matched Capacity'].sum())  # Use absolute sum for normalization
        relative_capacity = total_by_time / total_matched
        
        # Apply negative values for DK2->50HzT direction
        if border.upper() == 'DK2->50HZT':
            relative_capacity = -relative_capacity
            print(f"KPI 1.2 - Making values negative for {border}")
            print(f"Value range after negation: {relative_capacity.min():.4f} to {relative_capacity.max():.4f}")
        
        print(f"KPI 1.2 - {border} relative capacity range: {relative_capacity.min():.4f} to {relative_capacity.max():.4f}")
        
        # Track maximum absolute value for y-axis scaling
        max_rel_val = max(max_rel_val, abs(relative_capacity).max())
        
        # Add trace with custom styling
        is_dk2_to_50hzt = border.upper() == 'DK2->50HZT'
        line_color = 'rgb(31, 119, 180)' if is_dk2_to_50hzt else 'rgb(255, 127, 14)'
        trace_name = f"{border} ({'↓' if is_dk2_to_50hzt else '↑'})"
        
        fig.add_trace(go.Scatter(
            x=relative_capacity.index,
            y=relative_capacity.values,
            name=trace_name,
            mode='lines',
            line=dict(
                color=line_color,
                width=2
            ),
            hovertemplate='Time: %{x}<br>Direction: %{text}<br>Relative Capacity: %{y:.4f}<extra></extra>',
            text=[border] * len(relative_capacity)
        ))
    
    # Add 10% padding to the y-axis range
    max_abs_val = max_rel_val * 1.1
    print(f"\nKPI 1.2 - Y-axis range will be set to: -{max_abs_val:.4f} to {max_abs_val:.4f}")
    
    fig.update_layout(
        title=dict(
            text='KPI 1.2 - Relative Matched Capacity vs Time (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Relative Matched Capacity<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    return fig

def create_kpi_1_3(df):
    """KPI 1.3 - Matched capacity for implicit allocation per time to delivery start"""
    results = []
    fig = go.Figure()
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border]
        
        # Calculate total matched capacity for the entire period by delivery time
        delivery_sums = border_data.groupby('Delivery Start (CET)')['Matched Capacity'].sum()
        total_matched = delivery_sums.sum()
        
        # Group by both delivery and allocation time
        grouped_data = border_data.groupby(['Delivery Start (CET)', 'Allocation Time (CET)'])['Matched Capacity'].sum()
        
        # Apply negative sign for DK2->50HzT direction before calculating relative values
        if border.startswith('DK2'):
            grouped_data = -grouped_data
            
        # Calculate relative capacity over the entire period
        relative_capacity = grouped_data.groupby('Allocation Time (CET)').sum() / abs(total_matched)
        
        # Calculate average relative capacity
        avg_relative = relative_capacity
            
        fig.add_trace(go.Scatter(
            x=avg_relative.index,
            y=avg_relative.values,
            name=border,
            mode='lines',
            hovertemplate='Time: %{x}<br>Relative Capacity: %{y:.2f}<extra></extra>'
        ))
    
    # Calculate maximum absolute value for symmetric y-axis
    max_abs_val = max(
        abs(avg_relative.max()),
        abs(avg_relative.min())
    ) * 1.1  # Add 10% padding

    fig.update_layout(
        title=dict(
            text='KPI 1.3 - Relative Matched Capacity per Allocation Time (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Average Relative Matched Capacity<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    # Create statistics figure
    stats_data = pd.DataFrame({
        'Total Matched Capacity': delivery_sums,
        'Average Relative Capacity': avg_relative
    }).reset_index()
    
    stats_data = stats_data.round(4)
    
    fig_stats = go.Figure(data=[go.Table(
        header=dict(
            values=['Time', 'Total Matched Capacity', 'Average Relative Capacity'],
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[
                stats_data['index'].dt.strftime('%Y-%m-%d %H:%M:%S'),
                stats_data['Total Matched Capacity'],
                stats_data['Average Relative Capacity']
            ],
            fill_color='lavender',
            align='left'
        ))
    ])
    
    fig_stats.update_layout(
        title=f'KPI 1.3 - Statistics (50HzT↔DK2) - {border} Direction',
        height=800  # Make table taller to show more rows
    )
    
    results.append({
        'border': border,
        'main_fig': fig,
        'stats_fig': fig_stats
    })
    
    return results

def create_kpi_1_4(df):
    """KPI 1.4 - Matched capacity per implicit allocation per allocation time"""
    fig = go.Figure()
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border].copy()
        
        # Aggregate per 5 minutes
        allocation_time_grouped = border_data.groupby('Allocation Time (CET)')['Matched Capacity'].sum()
        
        # DK2->50HzT should be negative
        is_dk2_to_50hzt = border.upper() == 'DK2->50HZT'
        if is_dk2_to_50hzt:
            allocation_time_grouped = -allocation_time_grouped
            print(f"KPI 1.4 - Applying negative values for direction: {border}")
            print(f"Value range after negation: {allocation_time_grouped.min():.2f} to {allocation_time_grouped.max():.2f}")
        
        # Add trace with custom styling
        line_color = 'rgb(31, 119, 180)' if is_dk2_to_50hzt else 'rgb(255, 127, 14)'
        trace_name = f"{border} ({'↓' if is_dk2_to_50hzt else '↑'})"
        
        fig.add_trace(go.Scatter(
            x=allocation_time_grouped.index,
            y=allocation_time_grouped.values,
            name=trace_name,
            mode='lines',
            line=dict(
                color=line_color,
                width=2
            ),
            hovertemplate='Time: %{x}<br>Direction: %{text}<br>Capacity: %{y:.2f} MW<extra></extra>',
            text=[border] * len(allocation_time_grouped)
        ))
    
    # Calculate maximum absolute value for symmetric y-axis
    max_abs_val = max(
        abs(df['Matched Capacity'].max()),
        abs(df['Matched Capacity'].min())
    ) * 1.1  # Add 10% padding
    
    print(f"\nKPI 1.4 - Y-axis range will be set to: -{max_abs_val:.2f} to {max_abs_val:.2f}")
    
    fig.update_layout(
        title=dict(
            text='KPI 1.4 - Matched Capacity per Allocation Time (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Matched Capacity (MW)<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    return fig

def create_kpi_2(df):
    """KPI 2 - Available capacity per implicit allocation per border until delivery"""
    fig = go.Figure()
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border]
        
        # Group by allocation time
        atc_by_time = border_data.groupby('Allocation Time (CET)')['ATC with RR after Allocation'].mean()
        
        # DK2->50HzT should be negative
        if border.startswith('DK2'):
            atc_by_time = -atc_by_time
            
        fig.add_trace(go.Scatter(
            x=atc_by_time.index,
            y=atc_by_time.values,
            name=border,
            mode='lines',
            hovertemplate='Time: %{x}<br>ATC: %{y:.2f}<extra></extra>'
        ))
    
    # Calculate maximum absolute value for symmetric y-axis
    max_abs_val = max(
        abs(df['ATC with RR after Allocation'].max()),
        abs(df['ATC with RR after Allocation'].min())
    ) * 1.1  # Add 10% padding

    fig.update_layout(
        title=dict(
            text='KPI 2 - Available Capacity until Delivery (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Average ATC after Allocation (MW)<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    return fig

def create_kpi_3_1(df):
    """KPI 3.1 - When is max cross-border capacity for implicit allocations utilized"""
    fig = go.Figure()
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border]
        
        # Aggregate per 5 minutes using rounded allocation time
        delivery_grouped = border_data.groupby('Delivery Start (CET)').agg({
            'Matched Capacity': 'sum',
            'ATC with RR after Allocation': 'max'
        })
        
        # DK2->50HzT should be negative
        if border.startswith('DK2'):
            delivery_grouped['Matched Capacity'] = -delivery_grouped['Matched Capacity']
            delivery_grouped['ATC with RR after Allocation'] = -delivery_grouped['ATC with RR after Allocation']
        
        fig.add_trace(go.Scatter(
            x=delivery_grouped.index,
            y=delivery_grouped['Matched Capacity'],
            name=f'Total Matched Capacity ({border})',
            mode='lines',
            hovertemplate='Time: %{x}<br>Matched Capacity: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=delivery_grouped.index,
            y=delivery_grouped['ATC with RR after Allocation'],
            name=f'Max Available Capacity ({border})',
            mode='lines',
            line=dict(dash='dash'),
            hovertemplate='Time: %{x}<br>Available Capacity: %{y:.2f}<extra></extra>'
        ))
    
    # Calculate maximum absolute value for symmetric y-axis
    max_abs_val = max(
        abs(df['Matched Capacity'].max()),
        abs(df['ATC with RR after Allocation'].max())
    ) * 1.1  # Add 10% padding

    fig.update_layout(
        title=dict(
            text='KPI 3.1 - Capacity Utilization (50HzT↔DK2)',
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='Time',
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)',
            zerolinewidth=1
        ),
        yaxis=dict(
            title=dict(
                text='Capacity (MW)<br>↑ 50HzT→DK2 | ↓ DK2→50HzT',
                standoff=10
            ),
            gridcolor='rgba(0,0,0,0.1)',
            showgrid=True,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.5)',
            zerolinewidth=2,
            range=[-max_abs_val, max_abs_val]  # Symmetric range around zero
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=80, r=50, t=50, b=50)
    )
    
    return fig

def create_kpi_3_2(df):
    """KPI 3.2 - How often and how long is the full cross-border capacity utilized"""
    results = []
    
    # Create a single figure for both directions
    fig = go.Figure()
    all_stats = {}
    
    for border in df['Direction'].unique():
        border_data = df[df['Direction'] == border]
        
        # Apply negative values for DK2->50HzT direction before aggregation
        if border.startswith('DK2'):
            border_data['Matched Capacity'] = -border_data['Matched Capacity']
            border_data['ATC with RR after Allocation'] = -border_data['ATC with RR after Allocation']
            
        # Aggregate per hour
        hourly_data = border_data.groupby([
            pd.Grouper(key='Delivery Start (CET)', freq='1h')
        ]).agg({
            'Matched Capacity': 'sum',
            'ATC with RR after Allocation': 'max'
        }).reset_index()
        
        # Calculate utilization percentage
        hourly_data['Utilization_Percentage'] = (
            hourly_data['Matched Capacity'] / hourly_data['ATC with RR after Allocation'] * 100
        )
        
        # Consider capacity fully utilized if matched capacity is >= 95% of max capacity
        utilization_threshold = 0.95
        hourly_data['Fully_Utilized'] = (
            abs(hourly_data['Matched Capacity']) >= 
            utilization_threshold * abs(hourly_data['ATC with RR after Allocation'])
        )
        
        # Calculate statistics
        stats = {
            'Average Utilization (%)': hourly_data['Utilization_Percentage'].abs().mean(),
            'Max Utilization (%)': hourly_data['Utilization_Percentage'].abs().max(),
            'Hours Above 95%': hourly_data['Fully_Utilized'].sum(),
            'Hours Above 90%': (abs(hourly_data['Utilization_Percentage']) >= 90).sum(),
            'Hours Above 80%': (abs(hourly_data['Utilization_Percentage']) >= 80).sum(),
            'Average Matched Capacity': hourly_data['Matched Capacity'].abs().mean(),
            'Average Available Capacity': hourly_data['ATC with RR after Allocation'].abs().mean(),
        }
        all_stats[border] = stats
        
        # Calculate utilization percentage
        hourly_data['Utilization_Percentage'] = (
            hourly_data['Matched Capacity'] / hourly_data['ATC with RR after Allocation'] * 100
        )
        
        # Consider capacity fully utilized if matched capacity is >= 95% of max capacity
        utilization_threshold = 0.95
        hourly_data['Fully_Utilized'] = (
            hourly_data['Matched Capacity'] >= 
            utilization_threshold * hourly_data['ATC with RR after Allocation']
        )
        
        # Calculate statistics
        stats = {
            'Average Utilization (%)': hourly_data['Utilization_Percentage'].mean(),
            'Max Utilization (%)': hourly_data['Utilization_Percentage'].max(),
            'Hours Above 95%': hourly_data['Fully_Utilized'].sum(),
            'Hours Above 90%': (hourly_data['Utilization_Percentage'] >= 90).sum(),
            'Hours Above 80%': (hourly_data['Utilization_Percentage'] >= 80).sum(),
            'Average Matched Capacity': hourly_data['Matched Capacity'].mean(),
            'Average Available Capacity': hourly_data['ATC with RR after Allocation'].mean(),
        }
        
        # Add traces for this border
        fig.add_trace(
            go.Scatter(
                x=hourly_data['Delivery Start (CET)'],
                y=hourly_data['Utilization_Percentage'],
                name=f'Utilization Percentage ({border})',
                line=dict(color='blue' if not border.startswith('DK2') else 'lightblue')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_data['Delivery Start (CET)'],
                y=hourly_data['Matched Capacity'],
                name=f'Matched Capacity ({border})',
                line=dict(color='green' if not border.startswith('DK2') else 'lightgreen')
            )
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_data['Delivery Start (CET)'],
                y=hourly_data['ATC with RR after Allocation'],
                name=f'Maximum Capacity (ATC) ({border})',
                line=dict(color='red' if not border.startswith('DK2') else 'pink', dash='dash')
            )
        )
        
        # Add threshold line as a scatter trace
        fig.add_trace(
            go.Scatter(
                x=[hourly_data['Delivery Start (CET)'].min(), 
                   hourly_data['Delivery Start (CET)'].max()],
                y=[95, 95],
                name='95% Threshold',
                line=dict(color='red', dash='dash'),
                showlegend=True
            )
        )
        
        # Add statistics as annotations
        stats_text = "<br>".join([f"{k}: {v:.2f}" for k, v in stats.items()])
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            text=f"<b>Statistics:</b><br>{stats_text}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )
        
    # Add threshold lines
    fig.add_trace(
        go.Scatter(
            x=[hourly_data['Delivery Start (CET)'].min(), 
               hourly_data['Delivery Start (CET)'].max()],
            y=[95, 95],
            name='95% Threshold',
            line=dict(color='red', dash='dash'),
            showlegend=True
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[hourly_data['Delivery Start (CET)'].min(), 
               hourly_data['Delivery Start (CET)'].max()],
            y=[-95, -95],
            name='-95% Threshold',
            line=dict(color='red', dash='dash'),
            showlegend=False
        )
    )
    
    # Update layout
    fig.update_layout(
        title='KPI 3.2 - Capacity Utilization Analysis - Both Directions',
        xaxis_title='Time',
        yaxis_title='Capacity / Utilization %',
        showlegend=True,
        hovermode='x unified',
        # Add margin to accommodate statistics
        margin=dict(r=100, t=100, l=100)
    )
    
    # Create a separate figure for statistics table
    fig_stats = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Value'],
            fill_color='paleturquoise',
            align='left'
        ),
            cells=dict(
                values=[
                    list(stats.keys()),
                    [f"{v:.2f}" for v in stats.values()]
            ],
            fill_color='lavender',
            align='left'
        ))
    ])
    
    fig_stats.update_layout(
        title=f'KPI 3.2 - Utilization Statistics (50HzT↔DK2) - {border} Direction'
    )
    
    results.append({
            'border': border,
            'main_fig': fig,
            'stats_fig': fig_stats,
            'hourly_data': hourly_data,
            'statistics': stats
        })
    
    return results

def save_all_plots(figures, save_path):
    """Save all plots to a single HTML file"""
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("<html><head></head><body>\n")
        for title, fig in figures.items():
            f.write(f"<h2>{title}</h2>\n")
            # Handle both single figures and lists of figures
            if isinstance(fig, list):
                for subfig in fig:
                    if hasattr(subfig, 'to_html'):  # It's a figure
                        f.write(subfig.to_html(full_html=False, include_plotlyjs='cdn'))
                    elif isinstance(subfig, dict) and 'main_fig' in subfig:  # It's a dict with figures
                        f.write(subfig['main_fig'].to_html(full_html=False, include_plotlyjs='cdn'))
                        if 'stats_fig' in subfig:
                            f.write(subfig['stats_fig'].to_html(full_html=False, include_plotlyjs='cdn'))
            else:
                f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
            f.write("<hr>\n")
        f.write("</body></html>")

# File paths configuration - using absolute paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'Data', 'Allocations_June_10days.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(SCRIPT_DIR), 'output', 'all_kpi_plots.html')

def load_and_process_data():
    """Load and process the data from the CSV file."""
    df = load_data(INPUT_FILE)
    return df

def main():
    """Main function to process data and create visualizations"""
    print("Starting KPI Analysis...")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output file: {OUTPUT_FILE}")
    
    # Load and preprocess data
    print("1. Loading and preprocessing data...")
    df = load_and_process_data()
    
    print("2. Creating KPI visualizations...")
    print("   - Creating KPI 1.1...")
    kpi_1_1 = create_kpi_1_1(df)
    print("   - Creating KPI 1.2...")
    kpi_1_2 = create_kpi_1_2(df)
    print("   - Creating KPI 1.3...")
    kpi_1_3 = create_kpi_1_3(df)
    print("   - Creating KPI 1.4...")
    kpi_1_4 = create_kpi_1_4(df)
    print("   - Creating KPI 2...")
    kpi_2 = create_kpi_2(df)
    print("   - Creating KPI 3.1...")
    kpi_3_1 = create_kpi_3_1(df)
    print("   - Creating KPI 3.2...")
    kpi_3_2_results = create_kpi_3_2(df)
    
    # Create dictionary of figures
    figures = {
        'KPI 1.1 - Matched Capacity vs Time': kpi_1_1,
        'KPI 1.2 - Relative Matched Capacity vs Time': kpi_1_2,
        'KPI 1.3 - Relative Matched Capacity per Allocation Time': kpi_1_3,
        'KPI 1.4 - Matched Capacity per Allocation Time': kpi_1_4,
        'KPI 2 - Available Capacity until Delivery': kpi_2,
        'KPI 3.1 - Capacity Utilization': kpi_3_1
    }
    
    # Add KPI 3.2 figures
    for result in kpi_3_2_results:
        border = result['border']
        figures[f'KPI 3.2 - Capacity Utilization Analysis - {border}'] = result['main_fig']
        figures[f'KPI 3.2 - Utilization Statistics - {border}'] = result['stats_fig']
    
    # Save all plots to a single file
    save_all_plots(figures, OUTPUT_FILE)
    print(f"\nAnalysis complete! Results saved to: {OUTPUT_FILE}")
    
    return df

if __name__ == '__main__':
    main()