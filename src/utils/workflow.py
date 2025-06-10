import os
import numpy as np
import pandas as pd
import keras
import matplotlib.image as img
import tensorflow as tf

MODEL_CLASSIF_URI = 'https://storage.cloud.google.com/models-wagon1992-group-project/7/06ba2fef2e314aba94bdf6286c0440cc/artifacts/model/data/model.keras'
#MODEL_X_Y_URI =
#MODEL_Z_URI =

test_tomos = ['tomo_dae195', 'tomo_f2fa4a', 'tomo_cabaa0', 'tomo_f7f28b', 'tomo_ed1c97', 'tomo_ff505c', 'tomo_8f4d60', 'tomo_2aeb29', 'tomo_651ecd', 'tomo_e96200', 'tomo_0d4c9e', 'tomo_2dcd5c', 'tomo_983fce', 'tomo_7b1ee3', 'tomo_8b6795', 'tomo_dcb9b4', 'tomo_e764a7', 'tomo_e26c6b', 'tomo_331130', 'tomo_f8b835', 'tomo_746d88', 'tomo_9cd09e', 'tomo_b9eb9a', 'tomo_cf0875', 'tomo_7cf523', 'tomo_fd41c4', 'tomo_54e1a7', 'tomo_ca472a', 'tomo_6478e5', 'tomo_e9b7f2', 'tomo_247826', 'tomo_675583', 'tomo_f0adfc', 'tomo_378f43', 'tomo_19a313', 'tomo_172f08', 'tomo_f3e449', 'tomo_3b83c7', 'tomo_8c13d9', 'tomo_2c607f', 'tomo_c11e12', 'tomo_412d88', 'tomo_4b124b', 'tomo_38c2a6', 'tomo_ec1314', 'tomo_1c38fd', 'tomo_e63ab4', 'tomo_f07244', 'tomo_210371', 'tomo_d6e3c7', 'tomo_935f8a', 'tomo_a4c52f', 'tomo_a46b26', 'tomo_fadbe2', 'tomo_b28579', 'tomo_35ec84', 'tomo_369cce', 'tomo_6c203d', 'tomo_b80310', 'tomo_640a74', 'tomo_22976c', 'tomo_d21396', 'tomo_ecbc12', 'tomo_040b80', 'tomo_85708b', 'tomo_b98cf6', 'tomo_e1e5d3', 'tomo_138018', 'tomo_3264bc', 'tomo_e50f04', 'tomo_d723cd', 'tomo_2a6ca2', 'tomo_1f0e78', 'tomo_67565e', 'tomo_fd5b38', 'tomo_05b39c', 'tomo_372a5c', 'tomo_c3619a', 'tomo_ba76d8', 'tomo_a67e9f', 'tomo_a6646f', 'tomo_db656f', 'tomo_4102f1', 'tomo_bb5ac1', 'tomo_4ed9de', 'tomo_61e947', 'tomo_1da0da', 'tomo_821255', 'tomo_3e7783', 'tomo_c84b46', 'tomo_974fd4', 'tomo_444829', 'tomo_b50c0f', 'tomo_2a6091', 'tomo_fa5d78', 'tomo_bdd3a0', 'tomo_1c2534', 'tomo_d916dc', 'tomo_bdc097', 'tomo_7036ee', 'tomo_cacb75', 'tomo_5b359d', 'tomo_7fa3b1', 'tomo_049310', 'tomo_dd36c9', 'tomo_e3864f', 'tomo_0a8f05', 'tomo_ff7c20', 'tomo_0fab19', 'tomo_1c75ac', 'tomo_d0699e', 'tomo_1e9980', 'tomo_4ee35e', 'tomo_6943e6', 'tomo_99a3ce']

df = pd.read_csv('../data/csv_raw/train_labels.csv')

df_test = df[df['tomo_id'].isin(test_tomos)]

path = f'../data/pictures_process/adaptequal_1_padded'

X_test = [img.imread(f'{path}/{tomo}.jpg')/255 for tomo in test_tomos]
y_test = df_test.Number_of_motors


####### Motor Detection #######
classif_model = keras.saving.load_model('dl_gcp.keras') # changer le nom

preds = []

for X in X_test:
    X = X[:, :, 0]
    X = tf.expand_dims(X, axis=-1)
    X = tf.expand_dims(X, axis=0)
    preds.append(float(classif_model.predict(X)))

df_test['Motor_pred'] = preds

df_regression = df_test[df_test['Motor_pred'] >= 0.50]





####### Motor Position #######
### X, Y ###




### Z ###


####### Range check #######

####### Final fbetascore #######



for tomo_id in df_test.tomo_id:
    row_true = df_test[df_test['tomo_id'] == tomo_id].iloc[0]
    x_true, y_true, z_true = row_true['Motor_axis_0'], row_true['Motor_axis_1'], row_true['Motor_axis_2']
    row_pred = df_regression[df_regression['tomo_id'] == tomo_id].iloc[0]
    x_pred, y_pred, z_pred = row_true['Pred_motor_axis_0'], row_true['Pred_motor_axis_1'], row_true['Pred_motor_axis_2']
    euclid_distance = tf.norm(tf.Tensor(x_true, y_true, z_true)-tf.Tensor(x_pred, y_pred, z_pred), ord='euclidean')
    if euclid_distance > 3000:
        df_test[df_test['tomo_id'] == tomo_id]['Motor_pred'] = 0


from sklearn.metrics import fbeta_score



# test_tomos = ['tomo_dae195', 'tomo_f2fa4a', 'tomo_cabaa0', 'tomo_f7f28b', 'tomo_ed1c97', 'tomo_ff505c', 'tomo_8f4d60', 'tomo_2aeb29', 'tomo_651ecd', 'tomo_e96200', 'tomo_0d4c9e', 'tomo_2dcd5c', 'tomo_983fce', 'tomo_7b1ee3', 'tomo_8b6795', 'tomo_dcb9b4', 'tomo_e764a7', 'tomo_e26c6b', 'tomo_331130', 'tomo_f8b835', 'tomo_746d88', 'tomo_9cd09e', 'tomo_b9eb9a', 'tomo_cf0875', 'tomo_7cf523', 'tomo_fd41c4', 'tomo_54e1a7', 'tomo_ca472a', 'tomo_6478e5', 'tomo_e9b7f2', 'tomo_247826', 'tomo_675583', 'tomo_f0adfc', 'tomo_378f43', 'tomo_19a313', 'tomo_172f08', 'tomo_f3e449', 'tomo_3b83c7', 'tomo_8c13d9', 'tomo_2c607f', 'tomo_c11e12', 'tomo_412d88', 'tomo_4b124b', 'tomo_38c2a6', 'tomo_ec1314', 'tomo_1c38fd', 'tomo_e63ab4', 'tomo_f07244', 'tomo_210371', 'tomo_d6e3c7', 'tomo_935f8a', 'tomo_a4c52f', 'tomo_a46b26', 'tomo_fadbe2', 'tomo_b28579', 'tomo_35ec84', 'tomo_369cce', 'tomo_6c203d', 'tomo_b80310', 'tomo_640a74', 'tomo_22976c', 'tomo_d21396', 'tomo_ecbc12', 'tomo_040b80', 'tomo_85708b', 'tomo_b98cf6', 'tomo_e1e5d3', 'tomo_138018', 'tomo_3264bc', 'tomo_e50f04', 'tomo_d723cd', 'tomo_2a6ca2', 'tomo_1f0e78', 'tomo_67565e', 'tomo_fd5b38', 'tomo_05b39c', 'tomo_372a5c', 'tomo_c3619a', 'tomo_ba76d8', 'tomo_a67e9f', 'tomo_a6646f', 'tomo_db656f', 'tomo_4102f1', 'tomo_bb5ac1', 'tomo_4ed9de', 'tomo_61e947', 'tomo_1da0da', 'tomo_821255', 'tomo_3e7783', 'tomo_c84b46', 'tomo_974fd4', 'tomo_444829', 'tomo_b50c0f', 'tomo_2a6091', 'tomo_fa5d78', 'tomo_bdd3a0', 'tomo_1c2534', 'tomo_d916dc', 'tomo_bdc097', 'tomo_7036ee', 'tomo_cacb75', 'tomo_5b359d', 'tomo_7fa3b1', 'tomo_049310', 'tomo_dd36c9', 'tomo_e3864f', 'tomo_0a8f05', 'tomo_ff7c20', 'tomo_0fab19', 'tomo_1c75ac', 'tomo_d0699e', 'tomo_1e9980', 'tomo_4ee35e', 'tomo_6943e6', 'tomo_99a3ce']
# val_tomos = ['tomo_6f2c1f', 'tomo_dfc627', 'tomo_8d5995', 'tomo_cc2b5c', 'tomo_50cbd9', 'tomo_a72a52', 'tomo_9ae65f', 'tomo_9c0253', 'tomo_66285d', 'tomo_47d380', 'tomo_98686a', 'tomo_4077d8', 'tomo_97a2c6', 'tomo_ba9b3d', 'tomo_e2a336', 'tomo_aaa1fd', 'tomo_e8db69', 'tomo_532d49', 'tomo_f94504', 'tomo_5e2a91', 'tomo_2fc82d', 'tomo_16fce8', 'tomo_401341', 'tomo_0333fa', 'tomo_a81e01', 'tomo_b87c8e', 'tomo_e61cdf', 'tomo_b2ebbc', 'tomo_10c564', 'tomo_f71c16', 'tomo_47ac94', 'tomo_fea6e8', 'tomo_c00ab5', 'tomo_823bc7', 'tomo_278194', 'tomo_2fb12d', 'tomo_a537dd', 'tomo_19a4fd', 'tomo_417e5f', 'tomo_81445c', 'tomo_317656', 'tomo_7fbc49', 'tomo_806a8f', 'tomo_ab804d', 'tomo_957567', 'tomo_8634ee', 'tomo_fc1665', 'tomo_63e635', 'tomo_2645a0', 'tomo_5984bf', 'tomo_fc3c39', 'tomo_101279', 'tomo_08a6d6', 'tomo_0c2749', 'tomo_6607ec', 'tomo_23ce49', 'tomo_ca1d13', 'tomo_e55f81', 'tomo_bfd5ea', 'tomo_d7475d', 'tomo_136c8d', 'tomo_c4db00', 'tomo_ea3f3a', 'tomo_ef1a1a', 'tomo_2dd6bd', 'tomo_82d780', 'tomo_bede89', 'tomo_d5465a', 'tomo_e71210', 'tomo_9f1828', 'tomo_7550f4', 'tomo_efe1f8', 'tomo_bd42fa', 'tomo_01a877', 'tomo_59b470', 'tomo_0c3d78', 'tomo_d0c025', 'tomo_0eb41e', 'tomo_ca8be0', 'tomo_dbc66d', 'tomo_84997e', 'tomo_5dd63d', 'tomo_b9088c', 'tomo_24795a', 'tomo_6521dc', 'tomo_676744', 'tomo_cff77a', 'tomo_6f83d4', 'tomo_f78e91', 'tomo_6303f0', 'tomo_997437', 'tomo_cae587', 'tomo_9aee96', 'tomo_be9b98', 'tomo_97876d', 'tomo_e2da77', 'tomo_081a2d', 'tomo_cb5ec6', 'tomo_fc5ae4', 'tomo_4925ee', 'tomo_38d285', 'tomo_79a385', 'tomo_4469a7', 'tomo_05f919', 'tomo_568537', 'tomo_71ece1', 'tomo_85fa87', 'tomo_bcb115', 'tomo_2cace2', 'tomo_b4d92b', 'tomo_cc3fc4', 'tomo_94c173', 'tomo_a2a928', 'tomo_375513', 'tomo_40b215']
# train_tomos = ['tomo_c6f50a', 'tomo_288d4f', 'tomo_229f0a', 'tomo_decb81', 'tomo_39b15b', 'tomo_466489', 'tomo_d8c917', 'tomo_736dfa', 'tomo_03437b', 'tomo_066095', 'tomo_935ae0', 'tomo_c10f64', 'tomo_8e4919', 'tomo_2bb588', 'tomo_5bb31c', 'tomo_692081', 'tomo_ba37ec', 'tomo_20a9ed', 'tomo_b2b342', 'tomo_d396b5', 'tomo_b54396', 'tomo_122a02', 'tomo_4e3e37', 'tomo_9f918e', 'tomo_6cb0f0', 'tomo_f36495', 'tomo_e81143', 'tomo_6acb9e', 'tomo_56b9a3', 'tomo_dfdc32', 'tomo_98d455', 'tomo_2483bb', 'tomo_e5a091', 'tomo_91beab', 'tomo_b18127', 'tomo_73173f', 'tomo_221a47', 'tomo_5f1f0c', 'tomo_24a095', 'tomo_60d478', 'tomo_4f5a7b', 'tomo_975287', 'tomo_072a16', 'tomo_a8bf76', 'tomo_399bd9', 'tomo_512f98', 'tomo_4e38b8', 'tomo_146de2', 'tomo_423d52', 'tomo_711fad', 'tomo_b03f81', 'tomo_62dbea', 'tomo_a2bf30', 'tomo_25780f', 'tomo_8f5995', 'tomo_191bcb', 'tomo_372690', 'tomo_cf5bfc', 'tomo_f1bf2f', 'tomo_c678d9', 'tomo_cf53d0', 'tomo_c13fbf', 'tomo_e0739f', 'tomo_6d22d1', 'tomo_57c814', 'tomo_13973d', 'tomo_518a1f', 'tomo_dee783', 'tomo_556257', 'tomo_b11ddc', 'tomo_8e8368', 'tomo_e9fa5f', 'tomo_c649f8', 'tomo_517f70', 'tomo_622ca9', 'tomo_ac9fef', 'tomo_46250a', 'tomo_7eb641', 'tomo_bfdf19', 'tomo_6bb452', 'tomo_616f0b', 'tomo_f672c0', 'tomo_4c2e4e', 'tomo_9d3a0e', 'tomo_225d8f', 'tomo_d96d6e', 'tomo_5f34b3', 'tomo_93c0b4', 'tomo_464108', 'tomo_8554af', 'tomo_b24f1a', 'tomo_648adf', 'tomo_b7d94c', 'tomo_9ed470', 'tomo_db4517', 'tomo_3b8291', 'tomo_f427b3', 'tomo_f76529', 'tomo_5b087f', 'tomo_bc143f', 'tomo_774aae', 'tomo_b0e5c6', 'tomo_2f3261', 'tomo_b93a2d', 'tomo_ab78d0', 'tomo_c7b008', 'tomo_5764d6', 'tomo_2b996c', 'tomo_5308e8', 'tomo_e1a034', 'tomo_672101', 'tomo_0e9757', 'tomo_134bb0', 'tomo_643b20', 'tomo_d3bef7', 'tomo_f6de9b', 'tomo_8e58f1', 'tomo_4d528f', 'tomo_e72e60', 'tomo_0fe63f', 'tomo_a0cb00', 'tomo_221c8e', 'tomo_e57baf', 'tomo_02862f', 'tomo_738500', 'tomo_17143f', 'tomo_1446aa', 'tomo_0c3a99', 'tomo_8e4f7d', 'tomo_891afe', 'tomo_d0d9b6', 'tomo_aff073', 'tomo_0308c5', 'tomo_db2a10', 'tomo_d634b7', 'tomo_813916', 'tomo_30d4e5', 'tomo_6f0ee4', 'tomo_23c8a4', 'tomo_4f379f', 'tomo_37076e', 'tomo_37c426', 'tomo_fbb49b', 'tomo_e7c195', 'tomo_455dcd', 'tomo_fd9357', 'tomo_3eb9c8', 'tomo_cd1a7c', 'tomo_16136a', 'tomo_6a84b7', 'tomo_918e2b', 'tomo_c36b4b', 'tomo_f871ad', 'tomo_df866a', 'tomo_6733fa', 'tomo_3c6038', 'tomo_85edfd', 'tomo_4e41c2', 'tomo_9fc2b6', 'tomo_05df8a', 'tomo_7f0184', 'tomo_1cc887', 'tomo_06e11e', 'tomo_16efa8', 'tomo_8d231b', 'tomo_78b03d', 'tomo_32aaa7', 'tomo_1fb6a7', 'tomo_ac4f0d', 'tomo_971966', 'tomo_fb6ce6', 'tomo_50f0bf', 'tomo_285454', 'tomo_4b59a2', 'tomo_69d7c9', 'tomo_3183d2', 'tomo_2c9da1', 'tomo_d23087', 'tomo_3b7a22', 'tomo_4baff0', 'tomo_adc026', 'tomo_e5ac94', 'tomo_307f33', 'tomo_94a841', 'tomo_285d15', 'tomo_6bc974', 'tomo_51a47f', 'tomo_1dc5f9', 'tomo_7e3494', 'tomo_08bf73', 'tomo_0eb994', 'tomo_7dc063', 'tomo_79756f', 'tomo_6c5a26', 'tomo_76a42b', 'tomo_fb08b5', 'tomo_139d9e', 'tomo_e32b81', 'tomo_12f896', 'tomo_1af88d', 'tomo_603e40', 'tomo_f82a15', 'tomo_0f9df0', 'tomo_539259', 'tomo_fe85f6', 'tomo_fe050c', 'tomo_d31c96', 'tomo_c38e83', 'tomo_b10aa4', 'tomo_d83ff4', 'tomo_d5aa20', 'tomo_bbe766', 'tomo_305c97', 'tomo_5b8db4', 'tomo_5d798e', 'tomo_c8f3ce', 'tomo_881d84', 'tomo_da38ea', 'tomo_d0aa3b', 'tomo_abac2e', 'tomo_183270', 'tomo_51a77e', 'tomo_516cdd', 'tomo_c9d07c', 'tomo_656915', 'tomo_7dcfb8', 'tomo_a4f419', 'tomo_c4bfe2', 'tomo_eb4fd4', 'tomo_1ab322', 'tomo_6b1fd3', 'tomo_9dbc12', 'tomo_8e30f5', 'tomo_033ebe', 'tomo_a3ed10', 'tomo_4555b6', 'tomo_6e237a', 'tomo_5b34b2', 'tomo_3e6ead', 'tomo_891730', 'tomo_6df2d6', 'tomo_769126', 'tomo_95c0eb', 'tomo_bebadf', 'tomo_5f235a', 'tomo_3a3519', 'tomo_23a8e8', 'tomo_651ec2', 'tomo_3b1cc9', 'tomo_91c84c', 'tomo_a1a9a3', 'tomo_8ee8fd', 'tomo_28f9c1', 'tomo_67ff4e', 'tomo_acadd7', 'tomo_d2b1bc', 'tomo_53e048', 'tomo_a75c98', 'tomo_abb45a', 'tomo_493bea', 'tomo_0363f2', 'tomo_2c8ea2', 'tomo_30b580', 'tomo_a37a5c', 'tomo_569981', 'tomo_d84544', 'tomo_e685b8', 'tomo_ede779', 'tomo_b8595d', 'tomo_180bfd', 'tomo_4e478f', 'tomo_c596be', 'tomo_89d156', 'tomo_9f424e', 'tomo_71d2c0', 'tomo_646049', 'tomo_13484c', 'tomo_95e699', 'tomo_c36baf', 'tomo_9f222a', 'tomo_72b187', 'tomo_6a6a3b', 'tomo_91031e', 'tomo_abbd3b', 'tomo_7a9b64', 'tomo_868255', 'tomo_1efc28', 'tomo_319f79', 'tomo_256717', 'tomo_72763e', 'tomo_385eb6', 'tomo_3a0914', 'tomo_e34af8', 'tomo_4c1ca8', 'tomo_2acf68', 'tomo_24fda8', 'tomo_bad724', 'tomo_e51e5e', 'tomo_4e1b18', 'tomo_374ca7', 'tomo_9674bf', 'tomo_098751', 'tomo_c925ee', 'tomo_bde7f3', 'tomo_122c46', 'tomo_53c71b', 'tomo_e2ccab', 'tomo_68e123', 'tomo_9986f0', 'tomo_8e90f9', 'tomo_cc65a9', 'tomo_57592d', 'tomo_80bf0f', 'tomo_d9a2af', 'tomo_00e047', 'tomo_2a89bb', 'tomo_c4a4bb', 'tomo_0da370', 'tomo_6c4df3', 'tomo_47c399', 'tomo_9997b3', 'tomo_db6051', 'tomo_510f4e', 'tomo_60ddbd', 'tomo_bb9df3', 'tomo_79d622', 'tomo_499ee0', 'tomo_087d64', 'tomo_49f4ee', 'tomo_f8b46e', 'tomo_d2339b', 'tomo_2e1f4c', 'tomo_2c9f35', 'tomo_0a180f', 'tomo_a910fe', 'tomo_9cde9d', 'tomo_b7becf', 'tomo_d6c63f', 'tomo_be4a3a', 'tomo_c46d3c', 'tomo_161683', 'tomo_a020d7', 'tomo_b4d9da', 'tomo_513010', 'tomo_8f063a', 'tomo_b8f096', 'tomo_e77217', 'tomo_d26fcb']
