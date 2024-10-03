import torch
import torch.nn.functional as F
import torchaudio
import os
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from pathlib import Path
import csv
from tqdm import tqdm
import ast
import random
from SoundStream import audio_to_tokens
import re
from torchaudio.datasets import LibriLightLimited
import sys

myDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LibriDataset(Dataset):

    def __init__(self, audioDir, newSampleFreq, maxlength):
        self.audioList = self.getAudioList(audioDir)
        self.resampler = Resample(new_freq=newSampleFreq)
        self.maxlength = maxlength

    def getAudioList(self,audioDir):
        flac_files = []
        for root, dirs, files in os.walk(audioDir):
            for file in files:
                if file.endswith(".flac"):
                    file_path = os.path.join(root, file)
                    flac_files.append(file_path)
        return flac_files


    def __len__(self):
        return len(self.audioList)

    def __getitem__(self, idx):
        waveform, sampleRate = torchaudio.load(self.audioList[idx])
        waveformResampled = self.resampler(waveform)
        waveformPadded = F.pad(waveformResampled, pad = (0,self.maxlength - len(waveformResampled[0])))
        
        return waveformPadded

class TokensDataset(Dataset):

    def __init__(self, rootTokenDir, tokenFile, Q = 8, Q_prime = 3, sampleRate = 16000, mode = "semantic", removeSemanticDuplicates = True, row_limit = None, expected_audio_length = 60, crop_length = [30,10,3], removeOffset = False):

        self.validateParameters(rootTokenDir, tokenFile, mode, sampleRate)
        self.rootTokenDir = rootTokenDir
        self.tokenFile = tokenFile
        self.sampleRate = sampleRate
        self.mode = mode
        self.removeSemanticDuplicates = removeSemanticDuplicates
        self.row_limit = row_limit
        self.removeOffset = removeOffset

        # expressed in seconds, they define the expected duration of input audio and the crop length in the three different modes (semantic, coarse and  fine)
        
        self.expected_audio_length = expected_audio_length
        self.expected_crop_length = crop_length # CHANGE ME

        match mode:
        
            case "semantic":
                self.semanticlength = int(50 * self.expected_crop_length[0] - 1)
                self.coarselength = 0
                self.finelength = 0
                self.tokenTypeFlags = (False, True, False, False)
                self.num_samples = int(self.expected_audio_length / self.expected_crop_length[0])
                
            case "coarse":
                self.semanticlength = int(50 * self.expected_crop_length[1] - 1)
                self.coarselength = int((50 * self.expected_crop_length[1] + 1) * Q_prime) 
                self.finelength = 0
                self.tokenTypeFlags = (False, True, True, False)
                self.num_samples = int(self.expected_audio_length / self.expected_crop_length[1])
                
            case "fine":
                self.semanticlength = 0
                self.coarselength = int((50 * self.expected_crop_length[2] + 1) * Q_prime) 
                self.finelength = int((50 * self.expected_crop_length[2] + 1) * (Q - Q_prime))
                self.tokenTypeFlags = (False, False, True, True)
                self.num_samples = int(self.expected_audio_length / self.expected_crop_length[2])

            case _:
                raise ValueError('Invalid mode. Valid values are either "semantic", "coarse" or "fine".')
                
        self.inputs, self.labels = self.createTokenList()

    def validateParameters(self, rootTokenDir, tokenFile, mode, sampleRate):

        if not os.path.exists(os.path.join(rootTokenDir, tokenFile)):
            raise ValueError("Invalid rootTokenDir. It should be a valid directory path.")
        
        if not tokenFile.endswith('.csv'):
            raise ValueError("Invalid tokenFile. It should be a valid CSV file name.")
        
        if mode != "semantic" and mode != "coarse" and mode != "fine":
            raise ValueError("Invalid mode. It should be equal to semantic, coarse or fine")
        
        if not isinstance(sampleRate, int) or sampleRate <= 0:
            raise ValueError("Invalid sampleRate. It should be a positive integer.")

        
    def createTokenList(self):

        # Increase CSV row size limit
        prepare_csv_size()
        
        with open(os.path.join(self.rootTokenDir, self.tokenFile), mode='r', newline = '') as tokenFile:

            skipSep = False
            sep = tokenFile.readline().strip()
            match = re.match(r'sep=(.)', sep)
            if match:
                delimiter = match.group(1)
            else:
                skipSep = True
                delimiter = ";" #I assume this is the default

            reader = csv.reader(tokenFile, delimiter=delimiter)

            if not skipSep:
                sep = next(reader, None)

            header = next(reader, None)

            inputs = []
            labels = []

            row_counter = 0
            
            for row in reader:

                
                # stop when reached the row_limit, if set
                
                if self.row_limit and row_counter == self.row_limit:
                    return inputs, labels
                    
                row_counter += 1
                
                invalid_row = False
                formatted_row = [ast.literal_eval(cell) if isinstance(cell, str) and cell.startswith('[') and cell.endswith(']') else cell for cell in row]

                for j in range(self.num_samples):
                    invalid_row = False
                    sampled_row = []                    
                    for i, cell in enumerate(formatted_row):
                        if isinstance(cell, list):
                            match i:
                                case 1:
                                    N = self.semanticlength
                                case 2:
                                    N = self.coarselength
                                case 3:
                                    N = self.finelength
                                
                            if self.tokenTypeFlags[i]:
                                if len(cell) < (j + 1) * N:
                                    invalid_row = True
                                    break
                                    
                                if i == 1 and self.removeSemanticDuplicates:
                                    processedCell, _ = TokensDataset.__removeDuplicates(cell[j * N :(j + 1) * N])
                                else:
                                    processedCell = cell[j * N :(j + 1) * N]
                                    if self.removeOffset:
                                        processedCell = [elem % 1024 for elem in processedCell]
                                    
                                sampled_row.append(processedCell)

                    if not invalid_row:
                        if len(sampled_row) == 1:
                            inputs.append(torch.tensor(sampled_row[0][:-1]).to(myDevice))
                            labels.append(torch.tensor(sampled_row[0][1:]).to(myDevice))
                        else:
                            inputs.append(torch.tensor(sampled_row[0] + sampled_row[1][:-1]).to(myDevice))
                            labels.append(torch.tensor(sampled_row[1][1:]).to(myDevice))

        return inputs, labels 

    @staticmethod
    def __removeDuplicates(tokenList):
     
        PADDING_TOKEN = 0
        
        if not tokenList:
            return []
        
        processedList = [tokenList[0]]
        
        for i in range(1,len(tokenList)):
            if tokenList[i] != tokenList[i-1]:
                processedList.append(tokenList[i])
        
        paddinglength = len(tokenList) - len(processedList)
        processedList = processedList + [PADDING_TOKEN for i in range(paddinglength)]
        
        
        return processedList, paddinglength

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if idx >= len(self.inputs):
            raise IndexError("Index out of range")
        inputs = self.inputs[idx]
        labels = self.labels[idx]

        return inputs, labels

    def __str__(self):
        return f"self.rootTokenDir = {self.rootTokenDir}\n\
        self.tokenFile = {self.tokenFile}\n\
        self.sampleRate = {self.sampleRate}\n\
        self.removeSemanticDuplicates = {self.removeSemanticDuplicates}\n\
        self.row_limit = {self.row_limit}\n\
        self.expected_audio_length = {self.expected_audio_length}\n\
        self.expected_crop_length = {self.expected_crop_length}\n\
        self.semanticlength = {self.semanticlength}\n\
        self.coarselength = {self.coarselength}\n\
        self.finelength = {self.finelength}\n\
        self.num_samples = {self.num_samples}\n\
        "

def storeTokens(audioDir, outDir, outFile, w2vBERT, soundStream, fileCountCheckpoint = 5):

    Path(outDir).mkdir(parents=True, exist_ok=True)

    isNewFile = not os.path.exists(os.path.join(outDir, outFile))

    ## Check for eventual checkpoints
    fileChecked = 0
    reachedCheckpoint = False
    lastFile = None
    
    if os.path.exists(os.path.join(outDir, "checkpoint.txt")):
        with open(os.path.join(outDir, "checkpoint.txt"), mode='r', newline='') as checkpointFile:
            
            fileChecked, lastFile = checkpointFile.readline().strip().split(" ")
            fileChecked = int(fileChecked)
            print("Found a checkpoint!")

    tokenData = []
    fileCount = 0

    totalFiles = 0
    for root, dirs, files in os.walk(audioDir):
        totalFiles += len(files)

    with tqdm(total=totalFiles, desc='Processing files') as pbar:
        for root, dirs, files in os.walk(audioDir):
            for file in files:
    
                reachedCheckpoint = (fileChecked == 0 or file == lastFile or reachedCheckpoint)
                
                if file.endswith(".flac") and reachedCheckpoint and file != lastFile:
              
                    file_path = os.path.join(root, file)
                    waveform, sr = torchaudio.load(file_path)
                    with torch.no_grad():
                        semanticTokens, _ = w2vBERT(waveform)
                        coarseTokens, fineTokens = audio_to_tokens(waveform, soundStream)
                        
                    tokenData.append([file, semanticTokens.tolist(), coarseTokens.tolist(), fineTokens.tolist()])
    
                    fileCount += 1
    
                if fileCount % fileCountCheckpoint == 0 and reachedCheckpoint and file != lastFile:
                    with open(os.path.join(outDir, outFile), mode='a', newline='') as outFD, open(os.path.join(outDir, "checkpoint.txt"), mode='w', newline='') as checkpointFile:
                        writer = csv.writer(outFD, delimiter = ";")
    
                        ## Add header in case of newFile
                        if isNewFile:
                            outFD.write("sep=;\n")
                            writer.writerow(["fileName", "semanticTokens", "coarseTokens", "fineTokens"])
                            isNewFile = not isNewFile
                            
                        writer.writerows(tokenData)
    
                        checkpointFile.write(f"{fileCount + fileChecked} {file}")
                        
                    print(f"SAVED {fileCount} AUDIO ON OUTPUT {os.path.join(outDir, outFile)}. Total of {fileCount + fileChecked} records saved.") 
                    tokenData = []
                    
                pbar.update(1) 

    return fileCount

def prepare_single_audio(path, w2vBERT, soundStream, audioDuration, Q = 8, Q_prime = 3, removeOffset = False):

    semanticlength = int(50 * audioDuration - 1)
    coarselength = int((50 * audioDuration + 1) * Q_prime) 
    finelength = int((50 * audioDuration + 1) * (Q - Q_prime))

    waveform, sr = torchaudio.load(path)
    with torch.no_grad():
        semanticTokens, _ = w2vBERT(waveform)
        coarseTokens, fineTokens = audio_to_tokens(waveform, soundStream)
        
    semanticTokens = torch.tensor(semanticTokens)
    semanticTokens = semanticTokens[:semanticlength]
    coarseTokens = coarseTokens[:coarselength]
    fineTokens = fineTokens[:finelength]

    if removeOffset:
        semanticTokens = semanticTokens % 1024
        coarseTokens = coarseTokens % 1024
        fineTokens = fineTokens % 1024

    return semanticTokens, coarseTokens, fineTokens

def store_from_librilight(outDir, outFile, w2vBERT, soundStream, fileCountCheckpoint = 5, subset = "10h", lenght = None):
    Path(outDir).mkdir(parents=True, exist_ok=True)

    isNewFile = not os.path.exists(os.path.join(outDir, outFile))

    ## Check for eventual checkpoints
    fileChecked = 0
    reachedCheckpoint = False
    lastFile = None
    
    if os.path.exists(os.path.join(outDir, "checkpoint.txt")):
        with open(os.path.join(outDir, "checkpoint.txt"), mode='r', newline='') as checkpointFile:
            
            fileChecked, lastFileSP, lastFileCH, lastFileUT = checkpointFile.readline().strip().split(" ")
            fileChecked = int(fileChecked)
            lastFile = f"{lastFileSP} {lastFileSP} {lastFileUT}"
            print("Found a checkpoint!")

    tokenData = []
    fileCount = 0

    Path("./librilight").mkdir(parents=True, exist_ok=True)
    dataset = LibriLightLimited("./librilight", download=True, subset= subset)

    indices = list(range(len(dataset)))
    if lenght != None:
        random.seed(42)
        random.shuffle(indices)
        currentLenght = 0
        neededLenght = lenght * 60 * 60
    for i in indices:
        waveform, sr, transcript, spid, chid, utid = dataset[i]
        file = f"{spid} {chid} {utid}"
        reachedCheckpoint = (fileChecked == 0 or file == lastFile or reachedCheckpoint)
        
        if reachedCheckpoint and file != lastFile:
            waveform, sr = torchaudio.functional.resample(waveform, sr, 16000), 16000
            with torch.no_grad():
                semanticTokens, _ = w2vBERT(waveform)
                coarseTokens, fineTokens = audio_to_tokens(waveform, soundStream)
            
            if lenght != None:
                currentLenght += waveform.shape[-1] / sr

            tokenData.append([file, semanticTokens.tolist(), coarseTokens.tolist(), fineTokens.tolist()])

            fileCount += 1

        if (fileCount % fileCountCheckpoint == 0 and reachedCheckpoint and file != lastFile) or (lenght != None and currentLenght >= neededLenght):
            with open(os.path.join(outDir, outFile), mode='a', newline='') as outFD, open(os.path.join(outDir, "checkpoint.txt"), mode='w', newline='') as checkpointFile:
                writer = csv.writer(outFD, delimiter = ";")

                ## Add header in case of newFile
                if isNewFile:
                    outFD.write("sep=;\n")
                    writer.writerow(["fileName", "semanticTokens", "coarseTokens", "fineTokens"])
                    isNewFile = not isNewFile
                    
                writer.writerows(tokenData)

                checkpointFile.write(f"{fileCount + fileChecked} {file}")
                
            print(f"SAVED {fileCount} AUDIO ON OUTPUT {os.path.join(outDir, outFile)}. Total of {fileCount + fileChecked} records saved.") 
            tokenData = []
        if lenght != None and currentLenght >= neededLenght:
            break

    return fileCount

def prepare_csv_size():
    maxInt = sys.maxsize
    
    while True:
        # decrease the maxInt value by factor 10 
        # as long as the OverflowError occurs.
    
        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt/10)