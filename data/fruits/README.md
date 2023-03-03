## Fruits Dataset
This dataset was obtained from [here](). 

It should be in the following format in the repo: 
``` 
--od_pt_tutorial
    --data
        --fruits
            --train
            --valid
            --test
```

Upon initial download, the validation set will not exist. 
The `create_dataset.ipynb` notebook will create the train/test/validation split as well as 
create a single annotation CSV file for each dataset. 