public class Example {
    protected String name;
    protected int age;
    
    protected void displayInfo() {
        System.out.println("Name: " + name);
        System.out.println("Age: " + age);
    }
    
    protected String getName() {
        return name;
    }
}

class Child extends Example {
    public void setup() {
        name = "John";
        age = 25;
        displayInfo();
    }
}