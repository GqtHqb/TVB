import streamlit as st
from PIL import Image
from torchvision.transforms import v2

import torch

from utils import *

# # ————————————————————————————————————————————————————————————————————————————————————————————

st.set_page_config(layout='wide')

st.title(':rainbow[Position Predictor]')
            
col1, col2 = st.columns(2)

with col1:
    height = st.slider('Height (cm)', 130, 220, 180)
    weight = st.slider('Weight (kg)', 40, 150, 65)

    camera_on = st.toggle("Enable camera")
    img = st.camera_input('Bla', disabled=not camera_on, label_visibility='collapsed')


with col2:
    if img:
        pp = Preprocessing()

        img = Image.open(img)
        img = pp.preprocess_img(img)
        img = torch.tensor([img], dtype=torch.float32).unsqueeze(1)

        features = torch.tensor([(height, weight)], dtype=torch.float32)

        # Load model        
        model = CNNWithAdditionalFeatures()
        model.load_state_dict(torch.load('positionpredictor.pth', weights_only=False))

        model.eval()
        with torch.no_grad():
            prediction = model(img, features).detach()
        
        prediction_value = prediction.flatten()[0]
        prediction_label = pp.value2label(prediction_value)

        st.header(f'Prediction: **:rainbow[{prediction_label}]** ({prediction_value:.2f})')

        import plotly.figure_factory as ff

        bullet_data = [
            {
                "label": "", # Links van de horizontale bar
                "sublabel": "Top/Vers/Bottom",
                "range": [0.0, 0.0],
                "performance": [0.125, 0.375, 0.625, 0.875, 1.0],
                "point": [prediction_value],
            },
        ]

        # Create bullet chart
        fig = ff.create_bullet(
            bullet_data,
            titles='label',
            subtitles='sublabel',
            markers='point',
            measures='performance',
            ranges='range',
            orientation='h',
            title=f'Prediction: {prediction_label}'
        )
        
        # Customize markers
        fig.update_traces(
            marker=dict(
                color='purple',  # Change marker color
                size=15,      # Change marker size
                symbol='diamond'  # Change marker shape
            ),
            selector=dict(mode='markers')  # Apply only to markers
        )

        # Add annotation for the prediction point
        fig.add_annotation(
            x=prediction_value,  # Position on x-axis where the marker (point) is
            y=1,  # Centered on the y-axis
            text= f'{prediction_label} ({prediction_value:.2f})',  # Text to show
            showarrow=True,  # Arrow pointing to the marker
            arrowhead=2,  # Style of the arrow
            font=dict(size=15, color="black"),  # Font size and color
            align="center",
            arrowsize=1,
            arrowcolor="black",  # Color of the arrow
            ax=0,  # Arrow starting position (relative to the marker)
            ay=-30  # Vertical position of the text relative to the marker
        )

        # Add text below the x-axis at 0.0 and 1.0
        fig.add_annotation(
            x=0.0,  # Position on x-axis where the text will appear
            y=-0.05,  # A little below the x-axis
            text="Bottom",  # Text for the lower end
            showarrow=False,  # No arrow
            font=dict(size=12, color="black"),  # Font size and color
            align="center"
        )

        fig.add_annotation(
            x=1.0,  # Position on x-axis where the text will appear
            y=-0.05,  # A little below the x-axis
            text="Top",  # Text for the upper end
            showarrow=False,  # No arrow
            font=dict(size=12, color="black"),  # Font size and color
            align="center"
        )

        fig.add_annotation(
            x=0.5,  # Position on x-axis where the text will appear
            y=-0.05,  # A little below the x-axis
            text="Vers",  # Text for the upper end
            showarrow=False,  # No arrow
            font=dict(size=12, color="black"),  # Font size and color
            align="center"
        )

        # Update layout (adjust size)
        fig.update_layout(
            width=800,  # Set chart width
            height=300,  # Set chart height
            xaxis=dict(tickfont=dict(color="black")),  # Set x-axis tick values to black
            yaxis=dict(range=[-0.1, 1.1]),  # Extend y-axis range to make space for the text
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(
                l=0,  # Left margin
                # r=50,  # Right margin
                t=0,  # Top margin
                b=0   # Bottom margin
            ),  # Reduce margins to minimize empty space
            # title=dict(
            #     text=f'Prediction: {prediction_label}',  # Title text
            #     font=dict(size=16, color="black"),  # Set title font size and color
            #     x=0.5,  # Position title at the center
            #     xanchor="center"  # Anchor the title to the center
            # )
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig)
