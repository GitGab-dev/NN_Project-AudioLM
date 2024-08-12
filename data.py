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

class LibriDataset(Dataset):

    def __init__(self, audioDir, newSampleFreq, maxLenght):
        self.audioList = self.getAudioList(audioDir)
        self.resampler = Resample(new_freq=newSampleFreq)
        self.maxLenght = maxLenght

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
        waveformPadded = F.pad(waveformResampled, pad = (0,self.maxLenght - len(waveformResampled[0])))
        
        return waveformPadded

class TokensDataset(Dataset):

    def __init__(self, rootTokenDir, tokenFile, requiredDuration, Q = 8, Q_prime = 3, sampleRate = 16000, includeFileName = False, includeSemanticTokens = False, includeCoarseTokens = False, includeFineTokens = False):
        self.eosToken = -100
        self.validateParameters(rootTokenDir, tokenFile, requiredDuration, sampleRate)
        self.tokenTypeFlags = (includeFileName, includeSemanticTokens, includeCoarseTokens, includeFineTokens)
        self.rootTokenDir = rootTokenDir
        self.tokenFile = tokenFile
        match requiredDuration:
            case 3:
                self.semanticLenght = 150
                self.coarseLenght = int( 1208 * Q_prime / Q) 
                self.fineLenght = int( 1208 * (Q - Q_prime) / Q) 
            case 10:
                self.semanticLenght = 500
                self.coarseLenght = int( 4008 * Q_prime / Q) 
                self.fineLenght = int( 4008 * (Q - Q_prime) / Q) 
            case 30:
                self.semanticLenght = 1500
                self.coarseLenght = int( 12008 * Q_prime / Q) 
                self.fineLenght = int( 12008 * (Q - Q_prime) / Q) 
                
        self.sampleRate = sampleRate
        self.tokenList = self.createTokenList()

    def validateParameters(self, rootTokenDir, tokenFile, requiredDuration, sampleRate):

        if not os.path.exists(os.path.join(rootTokenDir, tokenFile)):
            raise ValueError("Invalid rootTokenDir. It should be a valid directory path.")
        
        if not tokenFile.endswith('.csv'):
            raise ValueError("Invalid tokenFile. It should be a valid CSV file name.")
        
        if not isinstance(requiredDuration, (int, float)) or requiredDuration <= 0:
            raise ValueError("Invalid requiredDuration. It should be a positive number.")
        
        if not isinstance(sampleRate, int) or sampleRate <= 0:
            raise ValueError("Invalid sampleRate. It should be a positive integer.")
        
    def createTokenList(self):
        
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

            data = []
            for row in reader:
                invalid_row = False
                formatted_row = [ast.literal_eval(cell) if isinstance(cell, str) and cell.startswith('[') and cell.endswith(']') else cell for cell in row]
                sampled_row = []
                for i, cell in enumerate(formatted_row):
                    if isinstance(cell, list):
                        match i:
                            case 1:
                                N = self.semanticLenght
                            case 2:
                                N = self.coarseLenght
                            case 3:
                                N = self.fineLenght
                            
                        if self.tokenTypeFlags[i]:
                            end_idx = len(cell) - N
                            if end_idx < 0:
                                invalid_row = True
                                break
                            start_idx = random.randint(0, end_idx)
                            sampled_row.append(cell[start_idx:start_idx + N])

                    elif self.tokenTypeFlags[i]:
                        sampled_row.append(cell)
                if not invalid_row:
                    data.append(sampled_row)

        return data

    def __len__(self):
        return len(self.tokenList)

    def __getitem__(self, idx):
        if idx >= len(self.tokenList):
            raise IndexError("Index out of range")

        input_tokens = torch.tensor(self.tokenList[idx][0]).unsqueeze(0)
        labels = torch.tensor(self.tokenList[idx][0][1:] + [self.eosToken]).unsqueeze(0)
        return input_tokens, labels


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


