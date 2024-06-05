import argparse

def get_option():
    parser = argparse.ArgumentParser(description="Training Parameter Setting")
    # model
    parser.add_argument("--ver", type=str, default="b3", help="base model")
    parser.add_argument("--pretrained", type=str, default=True, help="use or not use pretrained weight")   
    parser.add_argument("--loaded", type=str, default=False, help="load weight")   
    parser.add_argument("--load_path", type=str, default="", help="load weight")   

    # dir setting 
    parser.add_argument("--pth_dir", type=str, default="./checkpoint_save/", help="weights dir")   
    parser.add_argument("--data_dir", type=str, default="/hdd/Datasets/", help="train data dir")   
    parser.add_argument("--log_dir", type=str, default="./log/", help="log dir")   
    
    # optimizer setting
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")

    # data setting
    parser.add_argument("--img_size", type=int, default=512, help="train image crops size")  

    # training setting
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")  
    parser.add_argument("--total_epoch", type=int, default=50, help="training epochs") 
    parser.add_argument("--lr_init", type=float, default=1e-4, help="initilize learning rate")  
    parser.add_argument("--lr_min", type=float, default=0, help="minimum learning rate")   

    # 
    parser.add_argument("--is_resume", type=bool, default=False, help="is or not resume")  
    parser.add_argument("--device", type=str, default="cuda:0", help="calculate device")
    parser.add_argument("--resume_tag", type=str, default="last", help="resmue tag")

    args = parser.parse_args()
    return args 


def map_dict(args):
    assert type(args) == argparse.Namespace, "Input arguments error."
    args_dict = vars(args)
    return args_dict

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)
