# ---------------------------------------------------------------------
# ------------------------URL Scraper v.2.0.0-------------------------
# Created by: David Hodge
# Created on: 2020-8-17
# Description: This script is used for scraping the HTML code of an
#            input URL for web links. The HTML code of the site is
#            exported to a text file, and URL's are exported in a CSV
#            file.
# Version:
#       2.0.0 - Updated the CSVWriter class to include an object for
#            all links (including duplicates), and updated the
#            csvWrite method to write each link list to the CSV file
#            to a single column using the zip and zip_longest methods.
#            Also updated the URLScraper class to create a
#            BeautifulSoup object in the readURL method, use Prettify
#            to clean up the HTML text in the exportHTML method, and
#            created a list of all links (including duplicates) in the
#            findWebLinks method.
#
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------
#
# Script Modules:
import csv
import os
import urllib.request
from itertools import zip_longest
from bs4 import BeautifulSoup
from os import path
from urllib.parse import urlparse, urljoin
#
# Class: CSVWriter
# Summary:
#     Class CSVWriter writes objects to a CSV file
#
class CSVWriter:
#
# Constructor: __init__
    # Inputs:
    #     None
    # Processing:
    #     Constructor sets the inputs to their corresponding variables
    #     used in the class.
    # Output:
    #     None
#
    def __init__(self,
                 input1,
                 input2,
                 input3,
                 input4,
                 input5):
        self.internal = input1
        self.external = input2
        self.all = input3
        self.noDuplicates = input4
        self.csvName = input5
        self.csvInField = ["Internal Web Links (Duplicates Removed)"]
        self.csvExField = ["External Web Links (Duplicates Removed)"]
        self.csvAllField = ["All Web Links (Including Duplicates)"]
        self.csvNoField = ["All Web Links (Duplicates Removed)"]
#
# Method: csvWrite
    # Inputs:
    #     None
    # Processing:
    #     The csvWrite method writes the object values to a CSV file.
    # Output:
    #     None
#
    def csvWrite(self):
#
        print ("    Running the csvWrite method")
#
        # Determine whether or not CSV file already exists, if not
        # create new file.
        print ("        - Checking for existing CSV file")
        if os.path.isfile(self.csvName):
            # Delete original file
            print ("            - Deleting file...")
            os.remove(self.csvName)
#
        print ("        - Building CSV file")
        with open(self.csvName, 'w', newline="") as file:
            # Create CSV file, then populate it with fields and
            # values.
            print ("            - Creating file...")
            cFile = csv.writer(file,
                               delimiter=",",
                               quoting = csv.QUOTE_ALL)
            for row in zip(self.csvAllField,
                           self.csvNoField,
                           self.csvInField,
                           self.csvExField):
                cFile.writerow(row)
            for row in zip_longest(self.all,
                                   self.noDuplicates,
                                   self.internal,
                                   self.external):
                cFile.writerow(row)
#
# Class: URLScraper
# Summary:
#     Class URLScraper contains the object and functions used to
#     scrape the HTML code of a URL for web links.
#
class URLScraper:
#
# Constructor: __init__
    # Inputs:
    #     None
    # Processing:
    #     Constructor sets the inputs to their corresponding variables
    #     used in the class.
    # Output:
    #     None
#
    def __init__(self,
                 input,
                 name):
        self.url = input
        self.site = None
        self.fileName = name
        self.internalURL = set()
        self.externalURL = set()
        self.allURL = []
        self.noDuplicates = set()
        self.totalURLCount = 0
        self.totalDuplicateURLCount = 0
        self.externalURLCount = 0
        self.internalURLCount = 0
#
# Method: readURL
    # Inputs:
    #     None
    # Processing:
    #     The readURL method loads the URL input to an object.
    # Output:
    #     None
#
    def readURL(self):
#
        print ("    Running the readURL method")
#
        # Open the URL and convert the site HTML code to a sting
        print ("        - Reading URL")
        urlOpen = urllib.request.urlopen(self.url).read()
        print ("        - Opening URL in BeautifulSoup")
        self.site = BeautifulSoup(urlOpen,
                                  "html.parser")
        print ("")
#
# Method: exportHTML
    # Inputs:
    #     None
    # Processing:
    #     The exportURL method saves the HTML string to a text file.
    # Output:
    #     None
#
    def exportHTML(self):
#
        print ("    Running the exportHTML method")
#
        # Determine whether or not text file already exists
        print ("        - Checking for existing text file")
        if os.path.isfile(self.fileName):
            # Delete original file
            print ("            - Deleting file...")
            os.remove(self.fileName)
#
        # Create file, and then write HTML code to it
        print ("        - Creating text file")
        file = open(self.fileName,
                    "w+",
                    encoding="utf-8")
        file.write(self.site.prettify())
        file.close()
        print ("")
#
# Method: determineDomain
    # Inputs:
    #     None
    # Processing:
    #     The determineDomain method determines the domain of the URL.
    # Output:
    #     One output is returned to the calling object.
#
    def determineDomain(self):
#
        # Determine the URL domain
        domain = urlparse(self.url).netloc
        return domain
#
# Method: findWebLinks
    # Inputs:
    #     None
    # Processing:
    #     The findWebLinks method scrapes through the URL and locates
    #     all internal and external links.
    # Output:
    #     None
#
    def findWebLinks(self):
#
        print ("    Running the findWebLinks method")        
#
        # Call determineDomain method to find the domain of the URL.
        # This will be used to verify if a link is internal or
        # external.
        fD = self.determineDomain()
#
        # Search BeautifulSoup object for all HTML <a> tags.
        for link in self.site.findAll("a"):
            href = link.get("href")
#
            # Join the URL to the relative link and parse the link.
            # Break the URL into its components, then save the URL as
            # its scheme, net location, and path. Add the link to the
            # allURL object.
            href = urljoin(self.url, href)
            hrefParse = urlparse(href)
            hrefClean = (hrefParse.netloc + 
                        hrefParse.path)
            self.allURL.append(hrefClean)
            self.noDuplicates.add(hrefClean)
            self.totalURLCount += 1
#
            # Determine if the link is internal. If it is, check if it
            # is in the internalURL object, and then add it if it is
            # not.
            if fD in hrefClean:
                if hrefClean not in self.internalURL:
                    self.internalURL.add(hrefClean)
                    self.internalURLCount += 1
#
            # Determine if the link is external. If it is, check if it
            # is in the externalURL object, and then add it if it is
            # not.
            elif fD not in hrefClean:
                if hrefClean not in self.externalURL:
                    self.externalURL.add(hrefClean)
                    self.externalURLCount += 1
        self.totalDuplicateURLCount = (self.totalURLCount - 
                                        self.internalURLCount -
                                        self.externalURLCount)
        return self.internalURL, self.externalURL, self.allURL, self.noDuplicates;
#
# Method: printLinks
    # Inputs:
    #     None
    # Processing:
    #     The printLinks method prints the links in the internalURL and
    #     externalURL objects for testing purposes.
    # Output:
    #     None
#
    def printLinks(self):
        print ("")
        print ("The total internal link count is: {0:8}".format(
                                                     self.internalURLCount))
        print ("The total external link count is: {0:8}".format(
                                                     self.externalURLCount))
        print ("The total duplicate link count is: {0:7}".format(
                                                    self.totalDuplicateURLCount))
        print ("The total link count is: {0:17}".format(
                                                self.totalURLCount))
        print ("")
#
# Method: Main
# Inputs:
    #     None
    # Processing:
    #     The main method that runs the program.
    # Output:
    #     None
#
if __name__ == "__main__":
#
    print ("Beginning URLScraper program...")
    # Script variables
    urlInput = "http://www.census.gov/programs-surveys/popest.html"
    textOutput = "HTML_Export.txt"
    csvOutput = "URLScraper.csv"
#
    # Create instance of URLScraper class and call the readURL,
    # exportHTML, findWebLinks, and printLinks objects
    uS = URLScraper(urlInput,
                    textOutput)
    uS.readURL()
    uS.exportHTML()
    internalURL, externalURL, allURL, noDuplicates = uS.findWebLinks()
#
    # Create instance of CSVWriter class and call the csvWrite object
    cW = CSVWriter(internalURL,
                   externalURL,
                   allURL,
                   noDuplicates,
                   csvOutput)
    cW.csvWrite()
#
    uS.printLinks()