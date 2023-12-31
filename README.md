﻿# AnalyticsVidya

This project does classifies car images based on the damge recived to claim imsurance, and it's built with Python+Keras.

## Getting Started

Follow these steps to set up and run the project on your local machine.

### Prerequisites

- Python (version 3.11)
- Pip (package installer for Python)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/dhanushpandian/analyticsvidya.git
    cd analyticsvidya
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables:

    - Create a `.env` file in the root of the project.
    - Add your API keys and other sensitive information to the `.env` file. (Refer to [.env sample](#env-sample))

4. Run the project:

    ```bash
    python app.py
    ```

## Environment Variables

Store sensitive information like API keys in a `.env` file. Here's an example:

```plaintext
# .env file for HuggingFace

API_KEY=your_api_key_here

