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
    @classmethod
    def setUpClass(self):
        self.model = YOLO(path_to('runs', 'detect', '5_categories_no_aug', 'weights', 'best.pt'))

        image_paths = {
            'cat' : 'the_cat.png',
            'dog' : 'the_dog.jpg',
            'drone' : 'the_drone.jpg',
            'person' : 'the_person.png',
            'pikachu' : 'the_pikachu.jpeg',
        }
        self.images = {cat : path_to("test_images", image) for cat, image in image_paths.items()}

    def _test_category(self, category : str, confidence : float):
        detect_results = self.model.predict(self.images[category])

        detect_json = json.loads(detect_results[0].to_json())

        detect_category = detect_json[0]['name']
        detect_confidence = detect_json[0]['confidence']

        self.assertEqual(detect_category, category)
        self.assertGreater(detect_confidence, confidence)

    def test_cat(self):     self._test_category("cat",     0.75)

    def test_dog(self):     self._test_category("dog",     0.75)

    def test_drone(self):   self._test_category("drone",   0.75)

    def test_person(self):  self._test_category("person",  0.75)

    def test_pikachu(self): self._test_category("pikachu", 0.75)

if __name__ == "__main__":
    unittest.main()