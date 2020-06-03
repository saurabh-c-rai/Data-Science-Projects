# --------------
##File path for the file 
file_path 
def read_file(path):
    File = open(path,'r')
    sentence = File.readline()
    File.close()
    return sentence

sample_message = read_file(file_path)


#Code starts here


# --------------
#Code starts here
def read_file(path):
    File = open(path,'r')
    sentence = File.readline()
    File.close()
    return sentence

message_1= read_file(file_path_1)
message_2= read_file(file_path_2)
print(message_1,message_2)


def fuse_msg(message_a,message_b):
    quoitient = int(message_b)//int(message_a)
    return str(quoitient)

secret_msg_1 = fuse_msg(message_1,message_2)











# --------------
def read_file(path):
    File = open(path,'r')
    sentence = File.readline()
    File.close()
    return sentence

message_3= read_file(file_path_3)
print(message_3)

def substitute_msg(message_c):
    sub = ''
    if message_c == 'Red':
        sub = 'Army General'
        return sub
    elif message_c == 'Green':
        sub = 'Data Scientist'
        return sub
    elif message_c == 'Blue':
        sub = 'Marine Biologist' 
        return sub   
secret_msg_2 = substitute_msg(message_3) 










# --------------
# File path for message 4  and message 5
file_path_4
file_path_5

#Code starts here
def read_file(path):
    File = open(path,'r')
    sentence = File.readline()
    File.close()
    return sentence

message_4= read_file(file_path_4)
message_5= read_file(file_path_5)
print(message_4,message_5)

def compare_msg(message_d,message_e):
    a_list = message_d.split()
    b_list = message_e.split()
    c_list = []
    for i in a_list:
        if i not in b_list:
            c_list.append(i)
    final_msg = ' '.join(c_list)
    return final_msg


secret_msg_3 = compare_msg(message_4,message_5)











# --------------
#Code starts here
def read_file(path):
    File = open(path,'r')
    sentence = File.readline()
    File.close()
    return sentence

message_6= read_file(file_path_6)
print(message_6)

def extract_msg(message_f):
    a_list = message_f.split()

    even_word = (lambda x: True if len(x)%2==0 else False)
    b_list = filter(even_word,a_list)
    final_msg = ' '.join(b_list)
    return str(final_msg)

secret_msg_4 = extract_msg(message_6)


# --------------
#Secret message parts in the correct order
message_parts=[secret_msg_3, secret_msg_1, secret_msg_4, secret_msg_2]

final_path= user_data_dir + '/secret_message.txt'

#Code starts here 
a = str(message_parts[0]+ ' ' +message_parts[1]+' '+message_parts[2]+' '+message_parts[3])
b = a.replace('[','')
c = b.replace(']','')
secret_msg = c

def write_file(secret_msg,path):
    z = open(path , 'a+')
    x = z.write(secret_msg)
    z.close()
s = write_file(secret_msg,final_path)
print(secret_msg)




