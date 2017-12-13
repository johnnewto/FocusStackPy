import numpy as np
import timeit


setup = """
import numpy as np
A = np.ones((1000,1000,3), dtype=datatype)
"""

datatypes = "np.uint8", "np.uint16", "np.uint32", "np.uint64",  "np.float16", "np.float32", "np.float64"

stmt1 = """
A = A * 255
A = A / 255
A = A - 1
A = A + 1
"""
#~ np.uint8 : 1.04969205993
#~ np.uint16 : 1.19391073202
#~ np.uint32 : 1.37279821351
#~ np.uint64 : 2.99286961148
#~ np.float16 : 9.62375889588
#~ np.float32 : 0.884994368045
#~ np.float64 : 0.920502625252

stmt2 = """
A *= 255
A /= 255
A -= 1
A += 1
"""
#~ np.uint8 : 0.959514497259
#~ np.uint16 : 0.988570167659
#~ np.uint32 : 0.963571471946
#~ np.uint64 : 2.07768933333
#~ np.float16 : 9.40085450056
#~ np.float32 : 0.882363984225
#~ np.float64 : 0.910147440048

stmt3 = """
A = A * 255 / 255 - 1 + 1
"""
#~ np.uint8 : 1.05919667881
#~ np.uint16 : 1.20249978404
#~ np.uint32 : 1.58037744789
#~ np.uint64 : 3.47520357571
#~ np.float16 : 10.4792515701
#~ np.float32 : 1.29654744484
#~ np.float64 : 1.80735079168

stmt4 = """
A[:,:,:2] *= A[:,:,:2]
"""
#~ np.uint8 : 1.23270964172
#~ np.uint16 : 1.3260807837
#~ np.uint32 : 1.32571002402
#~ np.uint64 : 1.76836543305
#~ np.float16 : 2.83364821535
#~ np.float32 : 1.31282323872
#~ np.float64 : 1.44151875479

stmt5 = """
A[:,:,:2] = A[:,:,:2] * A[:,:,:2]
"""
#~ np.uint8 : 1.38166223494
#~ np.uint16 : 1.49569114821
#~ np.uint32 : 1.53105315419
#~ np.uint64 : 2.03457943366
#~ np.float16 : 3.01117795524
#~ np.float32 : 1.51807271679
#~ np.float64 : 1.7164808877

stmt6 = """
A *= 4
A /= 4
"""
#~ np.uint8 : 0.698176392658
#~ np.uint16 : 0.709560468038
#~ np.uint32 : 0.701653066443
#~ np.uint64 : 1.64199069295
#~ np.float16 : 4.86752675499
#~ np.float32 : 0.421001675475
#~ np.float64 : 0.433056710408

stmt7 = """
np.left_shift(A, 2, A)
np.right_shift(A, 2, A)
"""
#~ np.uint8 : 0.381521115341
#~ np.uint16 : 0.383545967785
#~ np.uint32 : 0.386147272415
#~ np.uint64 : 0.665969478824


for stmt in [stmt1, stmt2, stmt3, stmt4, stmt5, stmt6, stmt7]:
    print (stmt)
    for d in datatypes:
        s = setup.replace("datatype", d)
        T = timeit.Timer(stmt=stmt, setup=s)
        print (d,":", min(T.repeat(number=30)))
    print
print
