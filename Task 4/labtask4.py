card_number = [5, 8, 9, 3, 8, 0, 4, 1, 1, 5, 4, 5, 7, 2, 8, 9]
card_no = card_number[0 : -1]
card_no.reverse()

for items in range(len(card_no)):
    if items % 2 == 0:        
            card_no[items] *= 2
            if card_no[items] > 9:
                card_no[items] -= 9


total = sum(card_no) + 9
if total % 10 == 0:
     print("The number is valid")
else:
     print("Error!, The number is invalid!")

print(total)
