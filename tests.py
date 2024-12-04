import os
import json
import pydoc
import unittest
from os.path import join
from ultralytics import YOLO

def path_to(*paths : list[str]):    
    """
    Takes an arbitrary number of strings and converts it to a path
    with the current working directory automatically tacked onto the front of the path.
    
    Parameters:
        *p (str): Single string or list of strings.
    
    Returns:
        (string): Full path to desired directory/file
    """
    return join(os.getcwd(), *paths)

class SimpleTest(unittest.TestCase):
    # model under test
    MUT = YOLO(path_to('runs', 'detect', '5_categories_no_aug', 'weights', 'best.pt'))

    category_image = {
        'cat' : 'the_cat.png',
        'dog' : 'the_dog.png',
        'drone' : 'the_drone.jpg',
        'person' : 'the_person.png',
        'pikachu' : 'the_pikachu.jpeg',
    }
    category_image = {cat : path_to("test_images", image) for cat, image in category_image.items()}

    def test_category(self, category : str):
        detect_results = self.MUT.predict(self.category_image[category])

        json_category = json.loads(detect_results[0].to_json())
        
        detect_category = json_category[0]['name']

        self.assertEqual(detect_category, category)

    def test_cat(self):     self.test_category("cat")

    def test_dog(self):     self.test_category("dog")

    def test_drone(self):   self.test_category("drone")

    def test_person(self):  self.test_category("person")

    def test_pikachu(self): self.test_category("pikachu")

    def test_all_categories(self):
        for cat in self.category_image.keys():
            self.test_category(cat)

    # YOLO object creation successful

    # did not raise exception during .mp4 conversion

    # did not raise exception during live capture

if __name__ == "__main__":
    SimpleTest().test_all_categories()