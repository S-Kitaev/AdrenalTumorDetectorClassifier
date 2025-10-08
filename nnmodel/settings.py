settings = {
    'detection': {'all': 'media/nnModel/detectUZI/all/epoch93_P0,843_R0,741.pt'},
    'segmentation': {'all': 'media/nnModel/segUZI/all/Unet-timm-efficientnet-b7_dice-0.950.pt'},
    'classification': {
        'all':
            {
                'cv_models': [
                    'media/nnModel/classUZI/all/best90,5_s.pt',  # T2vsT3
                    'media/nnModel/classUZI/all/best95,1_l.pt',  # T2vsT4
                    'media/nnModel/classUZI/all/best90,6_m.pt',  # T2vsT5
                    'media/nnModel/classUZI/all/best77,1_n.pt',  # T3vsT4
                    'media/nnModel/classUZI/all/best84,9-l.pt',  # T3vsT5
                    'media/nnModel/classUZI/all/best77,5_s.pt',  # T4vsT5
                ],
                'ml_models': [
                    'media/nnModel/classUZI/all/XGB_without_US_type_73,13.pkl',
                    'media/nnModel/classUZI/all/XGB_with_US_type_73,88.pkl',
                    'media/nnModel/classUZI/all/XGB_ensemble_2_ml_models_74,63.pkl',
                ]
            }
    }
}