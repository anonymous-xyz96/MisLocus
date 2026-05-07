SubCellPortable
===============
This is a convenient code wrapper to run Lundberg lab SubCell model (created by Ankit Gupta) in inference mode with your own images.


Installation
------------

If you use any Python IDE (VSCode, PyCharm, Spyder, etc...), just:
- Either import the project into your IDE through git/github OR Create a new project and download all the repository code from github into it.
- Create a virtual environment for that project.
- Install the project requirements through your IDE.

If you want to install it via basic Python virtual environment:
- Install `python3`, `pip` and `virtualenv` in case you don't have them yet.
- Navigate to your desired working directory.
  - Example: `cd /home/lab/sandbox`
- Create a virtual environment:
  - Example: `python3 -m venv subcell`
- Download/cloe all the repository code inside your virtual environment directory.
- Navigate to your virtual environment directory and activate it:
  - Example: `source bin/activate`
- Install all requirements through pip:
  - Example: `pip install -r requirements.txt`
- Profit!


Setup
-----

You need to download the SubCell model files you want to use. The simplest way to do this is:

- Edit and modify `models_urls.yaml` file:
- Locate the lines related to the model(s) you want to use and input the model URLs.
  - So, for example, if you plan to run SubCell with 4 channels (nuclei, microtubules, ER and protein images) with "mae_contrast_supcon_model", you need to edit the 2 urls located under `rybg` group, `mae_contrast_supcon_model` subgroup.
- Use the `update_model` parameter set to `True` the fist time you run the model.

Of course, you can also bring the model files on your own: just make sure to place them into the proper folder structure under the `models` directory.


Running the code
---------------- 

**NOTE**: remember that you have to access your created virtual environment before running the code! If you are using an IDE you are probably ready to go, but if you have installed a basic python virtual environment remember to activate it like this: 
- Example:
   - `cd /home/lab/sandbox/example`
   - `source bin/activate`

To run SubCellPortable you have first to gather the information about the sets of images you want to process. SubCellPortable reads `path_list.csv` to locate each set of images, in the following .csv format: 

`r_image,y_image,b_image,g_image,output_folder,output_prefix`

- `r_image`: the microtubules targeting marker cell image. 
- `y_image`: the ER targeting marker cell image.
- `b_image`: the nuclei targeting marker cell image.
- `g_image`: the protein targeting marker cell image.
- `output_folder`: the base folder that will contain all results.
- `output_prefix`: the prefix appended to all files generated per cell.

All images can be relative or absolute paths, or directly URLs. You can also skip cells between runs with the special character `#` in front of the desired lines. 
Check the following `path_list.csv` content as an example of a possible run for the `rbg` model (microtubules, nuclei, protein):

```
#r_image,y_image,b_image,g_image,output_folder,output_prefix
images/cell_1_mt.png,,images/cell_1_nuc.png,images/cell_1_prot.png,output,cell1_
#images/cell_2_mt.png,,images/cell_2_nuc.png,images/cell_2_prot.png,output,cell2_
images/cell_3_mt.png,,images/cell_3_nuc.png,images/cell_3_prot.png,output,cell3_
```

Once you have prepared your `path_list.csv` you are ready to run the `process.py` script. You can choose between 3 different running approaches, depending on your personal preferences:

- Edit directly the constants located in the `process.py` script:
  - Probably the least versatile, but useful if you are always running SubCell with the same settings.
  - Just change the values under for the following section of code: `# If you want to use constants with your script, add them here` .
  - Simply call `python process.py`.

- Call `process.py` script with arguments:
  - You can get a list of available parameters (and their default values) using `-help` or `-?` argument.
  - Example call: `python process.py -c rbg -t mae_contrast_supcon_model -csv True`.

- Edit the `config.yaml` file:
  - Just change the contents of the file with your desired values.
  - Simply call `python process.py`.


Output
------ 

SubCell model creates the following items per each cell crop input:
- `[output_prefix]_embedding.npy`: the resulting vector embedding (1536 long) of the protein.
- `[output_prefix]_probabilities.npy`: an array of weighted probabilities of each subcellular location class.
- `[output_prefix]_attention_map`: a 64x64 PNG thumbnail of the attention map where the model has focused.

SubCellPortable provides some convenient information on top of that:
- In the `log.txt` produced by the run you can see the top 3 location names predicted. 
- If you use the `-csv` optional parameter a `result.csv` CSV file will be created with:
  - `id`: your chosen (hopefully unique) `output_prefix`.
  - `top_class_name,top_class,top_3_classes_names,top_3_classes`: convenient pre-calculated top predicted class and top 3 predicted classes.
  - `prob00-prob30`: full probabilities array.
  - `feat0000-feat1535`: full embedding vector.
- You can find the classes names in the `CLASS2NAME` dictionary at the beginning of the `inference.py` script.
- You can also find a helping `CLASS2COLOR` dictionary to assign a specific HEX color per class.