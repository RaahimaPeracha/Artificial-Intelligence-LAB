tasks= []

print("Create your To-Do List to help you in making your day productive")

while True:
 task = input("Enter your task: ")
 priority = int(input("Give your task a number according to its priority "))
 task_list = (task,priority)
 tasks.append(task_list)
 select = input("Do you want to add more tasks? (yes/no): ").lower()
 if select == 'no':
  break
 sort_tasks = sorted(tasks, key= lambda x: x[1])
 
 
print(tasks)