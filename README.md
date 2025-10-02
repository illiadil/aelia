# Aircraft Emergency Landing-site Identification and Assessment

A decision-support tool designed to identify and assess potential emergency landing sites for aircraft in real-time and during flight preparation.

## Description

This project provides a system for automatically identifying suitable emergency landing sites for aircraft experiencing critical in-flight situations. By analyzing geographical data, terrain elevation, surface conditions, and proximity to populated areas, the software calculates and ranks the safest possible landing options. The goal is to enhance aviation safety by providing pilots with crucial, data-driven information during an emergency.

-----

## Documentation 📄

For a detailed explanation of the user's manual, please refer to the full project documentation.

[ [**View Documentation.pdf**]](https://github.com/illiadil/aelia/blob/main/Documentation.pdf)

-----

## Installation

Follow these instructions to get the project running on your local machine.

1.  **Prerequisites**: Make sure you have **Miniconda** or **Anaconda** installed. This project is compatible with **Python 3.11.13+**. The `environment.yml` file will handle the installation of the correct Python version and dependencies.

2.  **Create Conda Environment**: Clone the repository and create the Conda environment.

    ```bash
    # Clone the repository
    git clone https://github.com/illiadil/aelia.git

    # Navigate to the project directory
    cd aelia

    # Create the conda environment using the environment.yml file
    conda env create -f environment.yml

    # Activate the new environment
    conda activate aeliatool
    ```

## Usage

Once your environment is activated, you can run the main script. Provide the aircraft's current coordinates, altitude, and heading as arguments.

  * **Running the analysis**:
    ```bash
    # Example for running the tool
    python source/mail.py
    ```
  * **Examples**:
      * The repository contains the "test images" which have 3 examples that can be used to test the tool. For each image, we associated it's corresponding top-left and bottom-right in a .txt file

-----

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. 🤝

1.  **Fork** the Project.
2.  Create your Feature Branch (`git checkout -b feature/NewFeature`).
3.  Commit your Changes (`git commit -m 'Add some NewFeature'`).
4.  Push to the Branch (`git push origin feature/NewFeature`).
5.  Open a **Pull Request**.

-----

## License

This project is distributed under the **MIT License**. See the `LICENSE.md` file for more information.

-----

## Contact

  * **Main contact** - [A. Illi](mailto:a.illi.ced@uca.ac.mam)
  * For bugs or feature requests, please **open an issue** on the GitHub repository.
