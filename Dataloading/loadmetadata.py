import pandas as pd

class MetadataLoader:
    def __init__(self, metadata_path):
        self.metadata_path = metadata_path
        self.metadata = None
    
    def load_metadata(self):
        if self.metadata is None:
            self.metadata = pd.read_csv(self.metadata_path)
        return self.metadata
    
    def get_disease_label(self, image_id):
        disease = self.metadata.loc[self.metadata['image_id'] == image_id, 'dx'].values
        return disease[0] if disease else None
 




