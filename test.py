# import numpy as np
# from sklearn.preprocessing import MinMaxScaler


# a = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
# s = MinMaxScaler()

# b = s.fit_transform(a)
# # print(b)
# # c = s.inverse_transform(b).reshape(-1)
# # print(c)

# # for i in range(0, len(b) - 2, 2):
# #     # print(b[i:i+2])
# #     b[i:i+2] = s.inverse_transform(b[i:i+2])
    
# # print(b)

# # print(s.inverse_transform(b[2].reshape(-1, 1)).reshape(-1))
# b[2:99] = s.inverse_transform(s.inverse_transform(b[2:99]))
# print(b)

# 100010
# 000010 
binary_str = "1101001"

# Encode the binary string as an integer
integer_value = int(binary_str, 2)
print(integer_value)
print(type(bin(integer_value)))
