"""
URL Feature Extraction Module for Phishing Detection

Extracts 41 features from raw URLs matching the dataset schema:
- URL structure features (length, components)
- Special character counts
- Domain and subdomain analysis
- Digit patterns
- Entropy calculations
"""

import re
from urllib.parse import urlparse
from collections import Counter
import math


class URLFeatureExtractor:
    """Extract features from raw URLs for phishing detection"""
    
    FEATURE_NAMES = [
        'url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url',
        'number_of_digits_in_url', 'number_of_special_char_in_url',
        'number_of_hyphens_in_url', 'number_of_underline_in_url',
        'number_of_slash_in_url', 'number_of_questionmark_in_url',
        'number_of_equal_in_url', 'number_of_at_in_url', 'number_of_dollar_in_url',
        'number_of_exclamation_in_url', 'number_of_hashtag_in_url',
        'number_of_percent_in_url', 'domain_length', 'number_of_dots_in_domain',
        'number_of_hyphens_in_domain', 'having_special_characters_in_domain',
        'number_of_special_characters_in_domain', 'having_digits_in_domain',
        'number_of_digits_in_domain', 'having_repeated_digits_in_domain',
        'number_of_subdomains', 'having_dot_in_subdomain',
        'having_hyphen_in_subdomain', 'average_subdomain_length',
        'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain',
        'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain',
        'having_digits_in_subdomain', 'number_of_digits_in_subdomain',
        'having_repeated_digits_in_subdomain', 'having_path', 'path_length',
        'having_query', 'having_fragment', 'having_anchor',
        'entropy_of_url', 'entropy_of_domain'
    ]
    
    def extract_features(self, url):
        """Extract all 41 features from a URL"""
        # Parse the URL
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path
        query = parsed.query
        fragment = parsed.fragment
        
        features = {}
        
        # URL-level features
        features['url_length'] = len(url)
        features['number_of_dots_in_url'] = url.count('.')
        features['having_repeated_digits_in_url'] = self._has_repeated_digits(url)
        features['number_of_digits_in_url'] = sum(c.isdigit() for c in url)
        
        # Special characters in URL
        special_chars = ['@', '?', '=', '$', '!', '#', '%', '-', '_', '/']
        features['number_of_special_char_in_url'] = sum(url.count(c) for c in special_chars)
        features['number_of_hyphens_in_url'] = url.count('-')
        features['number_of_underline_in_url'] = url.count('_')
        features['number_of_slash_in_url'] = url.count('/')
        features['number_of_questionmark_in_url'] = url.count('?')
        features['number_of_equal_in_url'] = url.count('=')
        features['number_of_at_in_url'] = url.count('@')
        features['number_of_dollar_in_url'] = url.count('$')
        features['number_of_exclamation_in_url'] = url.count('!')
        features['number_of_hashtag_in_url'] = url.count('#')
        features['number_of_percent_in_url'] = url.count('%')
        
        # Domain features
        features['domain_length'] = len(domain)
        features['number_of_dots_in_domain'] = domain.count('.')
        features['number_of_hyphens_in_domain'] = domain.count('-')
        
        domain_special_chars = ['@', '-', '_']
        features['having_special_characters_in_domain'] = int(any(c in domain for c in domain_special_chars))
        features['number_of_special_characters_in_domain'] = sum(domain.count(c) for c in domain_special_chars)
        
        features['having_digits_in_domain'] = int(any(c.isdigit() for c in domain))
        features['number_of_digits_in_domain'] = sum(c.isdigit() for c in domain)
        features['having_repeated_digits_in_domain'] = self._has_repeated_digits(domain)
        
        # Subdomain features
        subdomains = self._extract_subdomains(domain)
        features['number_of_subdomains'] = len(subdomains)
        
        if subdomains:
            features['having_dot_in_subdomain'] = int(any('.' in sd for sd in subdomains))
            features['having_hyphen_in_subdomain'] = int(any('-' in sd for sd in subdomains))
            features['average_subdomain_length'] = sum(len(sd) for sd in subdomains) / len(subdomains)
            features['average_number_of_dots_in_subdomain'] = sum(sd.count('.') for sd in subdomains) / len(subdomains)
            features['average_number_of_hyphens_in_subdomain'] = sum(sd.count('-') for sd in subdomains) / len(subdomains)
            features['having_special_characters_in_subdomain'] = int(any(any(c in sd for c in domain_special_chars) for sd in subdomains))
            features['number_of_special_characters_in_subdomain'] = sum(sum(sd.count(c) for c in domain_special_chars) for sd in subdomains)
            features['having_digits_in_subdomain'] = int(any(any(c.isdigit() for c in sd) for sd in subdomains))
            features['number_of_digits_in_subdomain'] = sum(sum(c.isdigit() for c in sd) for sd in subdomains)
            features['having_repeated_digits_in_subdomain'] = int(any(self._has_repeated_digits(sd) for sd in subdomains))
        else:
            # Default values for empty subdomains
            features['having_dot_in_subdomain'] = 0
            features['having_hyphen_in_subdomain'] = 0
            features['average_subdomain_length'] = 3
            features['average_number_of_dots_in_subdomain'] = 0
            features['average_number_of_hyphens_in_subdomain'] = 0
            features['having_special_characters_in_subdomain'] = 1
            features['number_of_special_characters_in_subdomain'] = 3
            features['having_digits_in_subdomain'] = 0
            features['number_of_digits_in_subdomain'] = 0
            features['having_repeated_digits_in_subdomain'] = 1
        
        # Path, query, fragment features
        features['having_path'] = int(len(path) > 1)  # path always has at least '/'
        features['path_length'] = len(path) - 1 if len(path) > 1 else 0  # exclude leading '/'
        features['having_query'] = int(len(query) > 0)
        features['having_fragment'] = int(len(fragment) > 0)
        features['having_anchor'] = int('#' in url)
        
        # Entropy calculations
        features['entropy_of_url'] = self._calculate_entropy(url)
        features['entropy_of_domain'] = self._calculate_entropy(domain)
        
        return features
    
    def _has_repeated_digits(self, text):
        """Check if there are consecutive repeated digits"""
        for i in range(len(text) - 1):
            if text[i].isdigit() and text[i] == text[i + 1]:
                return 1
        return 0
    
    def _extract_subdomains(self, domain):
        """Extract subdomains from domain"""
        if not domain:
            return []
        
        parts = domain.split('.')
        # Typically, the last two parts are the main domain (e.g., example.com)
        # Everything before that is considered subdomains
        if len(parts) > 2:
            return parts[:-2]
        elif len(parts) == 2:
            return parts[:1]  # Return the first part as subdomain
        return []
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of text"""
        if not text:
            return 0.0
        
        # Count character frequencies
        counter = Counter(text)
        length = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
        
        return entropy
    
    def extract_as_array(self, url):
        """Extract features and return as ordered array matching dataset schema"""
        features = self.extract_features(url)
        return [features[name] for name in self.FEATURE_NAMES]
    
    def extract_batch(self, urls):
        """Extract features from multiple URLs"""
        return [self.extract_as_array(url) for url in urls]


# Example usage
if __name__ == "__main__":
    extractor = URLFeatureExtractor()
    
    # Test with sample URLs
    test_urls = [
        "https://www.google.com",
        "http://phishing-example-123.com/login?user=test",
        "https://secure.bank.example.com/account/dashboard"
    ]
    
    for url in test_urls:
        features = extractor.extract_features(url)
        print(f"\nURL: {url}")
        print(f"Features: {len(features)} extracted")
        print(f"URL Length: {features['url_length']}")
        print(f"Domain: {features['domain_length']} chars")
        print(f"Entropy: {features['entropy_of_url']:.3f}")
