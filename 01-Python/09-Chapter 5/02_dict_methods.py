marks = {
    "Prathamesh": 100,
    "Shubham": 56,
    "Rohan": 23,
    0: "Prathamesh"
}

# print(marks.items())
# print(marks.keys())
# print(marks.values())
# marks.update({"Prathamesh": 99, "Renuka": 100})
# print(marks)

print(marks.get("Prathamesh2")) # Prints None
print(marks["Prathamesh2"]) # Returns an error