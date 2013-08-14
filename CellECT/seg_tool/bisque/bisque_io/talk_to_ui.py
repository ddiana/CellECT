# Imports
from bq.api.comm import BQSession
from xml import etree
import logging
import time

# Imports from this project
import CellECT.seg_tool.globals 

def wait_for_UI_update():

	""" Updates status to bisque mex (ready to take user input).
		Polls the mex repeatedly until the UI has posted updated status (ready for more processing).
	"""

	# get mex information
	mex_url = CellECT.seg_tool.globals.DEFAULT_PARAMETER["bq_mex_url"]
	access_token = CellECT.seg_tool.globals.DEFAULT_PARAMETER["bq_token"]

	# start bisque session
	bqsession = BQSession()
	bqsession.init_mex(mex_url, access_token)

	# post "ready for UI"
	bqsession.update_mex("READY_FOR_INTERACTION")
	logging.info ("Passed to UI.")
			
	# retrieve mex and check status
	mex_etree = bqsession.fetchxml(mex_url+'?view=short')
	status = mex_etree.attrib['value']

	logging.info("Waiting for UI update.")
	
	# wait for UI
			
	while status != "READY_FOR_PROCESSING":

		print "Waiting for UI..."
		mex_etree = bqsession.fetchxml(mex_url+'?view=short')
		status = mex_etree.attrib['value']
		#print status

		######### REMOVE THIS WHEN UI IS DONE
		bqsession.update_mex("READY_FOR_PROCESSING")

		time.sleep(2)


	# UI posted update.
	logging.info("Getting update from UI.")
	bqsession.update_mex("Processing user feedback...")
	print "FINISHED"

